#=
    A wrapper to simplify gmsh interactions. 
    It helps users with:
        . Create simple geometries
        . Combine them into a unified model
        . Create a local refinement field using a single scalar value (localSize)
        . Convert the gmsh mesh to a more user friendly MESH() struct
        . Make a bounding shell for open boundary problems
        . Manage surface triangles and surface ID's
=#

function save2file(fileName,input)
    # Saves matrix to a .txt file
    open(fileName, "w") do io
        for row in eachrow(input)
            println(io, join(row, " , "))  # Space-separated
        end
    end
end # Save matrix to .txt file

# Local mesh refinement on target cell
function refineCell(cell,localSize,meshSize)
    #=
        Sets every volume in 'cell' to be locally refined with target 'localSize' 
    =#

    # Get the boundary of the cell 
    cell_boundary = gmsh.model.getBoundary(cell, false, false, false)

    # Create a distance field for local refinement
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", [s[2] for s in cell_boundary])

    # Create a threshold field that defines the refinement region
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", localSize)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", meshSize)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)

    # Use the minimum of all the fields as the background mesh field
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])

    # Set the background mesh field
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
end # Local mesh refinement on target cell

# Make a cuboid based on its center
function addCuboid(position,dimensions,cells=[],updateCells=false)
    #=
        Makes a cuboid based on its centroid position
        Updates the cells list in case this cuboid is not meant to be
        the container of the simulation
    =#

    r = position - dimensions/2
    box = gmsh.model.occ.addBox(r[1], r[2], r[3], dimensions[1], dimensions[2], dimensions[3])
    
    if updateCells
        cells = append!(cells,[(3,box)])
    end

    return box
end # Make a cuboid based on its center

# Make a sphere
function addSphere(position,radius,cells=[],updateCells=true)
    #=
        Inputs:
            Position vector
            radius value
            cells <- a list of volumes (cells) that are inside a container
    =# 
    
    # Add a sphere to the current model
    sphere = gmsh.model.occ.addSphere(position[1],position[2],position[3],radius)

    # If sphere is not the container
    if updateCells    
        cells = append!(cells,[(3,sphere)])
    end

    return sphere
end # Make a sphere

# Create container based on current model surface
function makeContainer(scale=5)

    # Get all surface entities
    surfaces = gmsh.model.getEntities(2)

    # Initialize min/max coordinates
    x_min = Inf
    y_min = Inf
    z_min = Inf
    x_max = -Inf
    y_max = -Inf
    z_max = -Inf

    # Find global bounding box of the STL
    for s in surfaces
        bb = gmsh.model.getBoundingBox(s[1], s[2])
        x_min = min(x_min, bb[1])
        y_min = min(y_min, bb[2])
        z_min = min(z_min, bb[3])
        x_max = max(x_max, bb[4])
        y_max = max(y_max, bb[5])
        z_max = max(z_max, bb[6])
    end

    Lx = x_max-x_min
    Ly = y_max-y_min
    Lz = z_max-z_min
    L = maximum([Lx,Ly,Lz])

    # Container position and dimensions
    center = [(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2]
    dimensions = scale*[L,L,L]

    box = addCuboid(center,dimensions)

    # Update model
    gmsh.model.occ.synchronize()

    return box
end # Create container based on current model surface

# Import cad geometry file
function importCAD(file,cells=[],box=[],scale=5)
    #= 
        Import cad geometry file and create a container
        if there is none
    =#

    # Import CAD file (BREP, STEP or IGES)
    gmsh.model.occ.importShapes(file)
    volume = gmsh.model.occ.healShapes()
    _,volume = volume[1]

    gmsh.model.occ.synchronize()
    cells = append!(cells,[(3,volume)])

    # Make a container for the stl file
    if isempty(box)
        box = makeContainer(scale)
        gmsh.model.geo.synchronize()
    end

    return box
end # Import cad geometry file

function Mesh(cells,meshSize=0,localSize=0,saveMesh=false)
    #=
        Generates a 3d tetrahedral mesh considering that the model is made of 
        1 container and every other volume beyond the container is listed in the 'cells'

        Inputs
            cells       -> geometries that are inside the container
            meshSize    -> overall target mesh size
            localSize   -> Local mesh refinement
    =#

    # Make a mesh object
    mesh = MESH()

    # Get the volume IDs of the cells inside the container
    volumeID = []
    for i in cells
        append!(volumeID,i[2])
    end

    # >> Mesh settings
    
    # Set local mesh size
    if localSize>0
        refineCell(cells,localSize,meshSize) # Set local refinement on the sphere Cell
    end

    # Set maximum element size
    if meshSize > 0
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshSize)
    end

    # Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen") # Optimize the mesh

    # Get all tetrahedral elements (4 - tetrahedrons)
    t_tags, t = gmsh.model.mesh.getElementsByType(4)
    mesh.t = reshape(t,4,Int(size(t,1)/4))
    mesh.nt = size(mesh.t,2)

    # Get node coordinates
    _,p,_ = gmsh.model.mesh.getNodes()
    mesh.p = reshape(p, 3, Int(size(p,1)/3))
    mesh.nv = size(mesh.p,2)

    # Get all surface triangles
    surfaceT_tags, surfaceT = gmsh.model.mesh.getElementsByType(2)
    surfaceT = reshape(surfaceT, 3, Int(size(surfaceT,1)/3))

    # Expand surface triangles to include boundary id
    surfaceT = [surfaceT;UInt.(zeros(1,size(surfaceT,2)))]
    for i in 1:size(surfaceT,2)
        # Get ID of the element of the current surface triangle
        _,_,_, id = gmsh.model.mesh.getElement(surfaceT_tags[i])
        surfaceT[end,i] = id; # Set the surface triangle boundary id
    end

    mesh.surfaceT = surfaceT
    mesh.ne = size(mesh.surfaceT,2)

    # Mesh elements inside the container
    InsideElements = zeros(mesh.nt,1)
    for k in 1:mesh.nt
        etype,nodeTags,dim, id = gmsh.model.mesh.getElement(t_tags[k])
        # element type , nodes of the element , dimension , id
        if id in volumeID
            InsideElements[k] = k
        end
    end
    mesh.InsideElements = Int.(InsideElements[InsideElements.!=0])
    mesh.nInside = length(mesh.InsideElements)

    # Inside nodes
    aux::Matrix{Int32} = mesh.t[:,mesh.InsideElements]
    mesh.InsideNodes = unique(vec(aux))

    # Element volumes
    mesh.VE = zeros(mesh.nt)
    for k in 1:mesh.nt
        mesh.VE[k] = elementVolume(mesh.p,mesh.t[:,k])
    end

    # List of all surface triangle normals
    mesh.normal = zeros(3,mesh.ne);
    for i in 1:mesh.ne
        mesh.normal[:,i] = normal_surface(mesh.p,@view mesh.surfaceT[1:3,i]);
    end

    # Save mesh 
    if saveMesh
        save2file("t.txt",mesh.t) # Save connectivity list to a .txt file
        save2file("p.txt",mesh.p)
        save2file("InsideNodes.txt",mesh.InsideNodes)
        save2file("surfaceT.txt",mesh.surfaceT)
        save2file("InsideElements.txt",mesh.InsideElements)
        save2file("VE.txt",mesh.VE)
        save2file("normal.txt",mesh.normal)
    end

    return mesh
end

# Normal to surface triangle
function normal_surface(p,nds)
    # Reshape coords into 3 points (x,y,z)
    p1 = p[:,nds[1]]
    p2 = p[:,nds[2]]
    p3 = p[:,nds[3]]
    # Edge vectors
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product (normal vector)
    n = [v1[2]*v2[3] - v1[3]*v2[2],
         v1[3]*v2[1] - v1[1]*v2[3],
         v1[1]*v2[2] - v1[2]*v2[1]]
    # Normalize
    norm_n = sqrt(n[1]^2 + n[2]^2 + n[3]^2)
    return n ./ norm_n
end # Normal to surface triangle

# Area of the 3D triangle
function areaTriangle(xt,yt,zt)
    Atr = 0.5*sqrt(det([xt';yt';[1 1 1]])^2 + det([yt';zt';[1 1 1]])^2 + det([zt';xt';[1 1 1]])^2);
    return Atr
end # Area of the 3D triangle

# Mesh element volume
function elementVolume(p,nds)
    # Extract the four nodes (columns of p)
    A = p[:, nds[1]]
    B = p[:, nds[2]]
    C = p[:, nds[3]]
    D = p[:, nds[4]]

    # Compute vectors AB, AC, AD
    AB = B - A
    AC = C - A
    AD = D - A

    # Compute the scalar triple product (AB ⋅ (AC × AD))
    cross_AC_AD = [AC[2]*AD[3] - AC[3]*AD[2],
                   AC[3]*AD[1] - AC[1]*AD[3],
                   AC[1]*AD[2] - AC[2]*AD[1]]
    triple_product = AB[1] * cross_AC_AD[1] + AB[2] * cross_AC_AD[2] + AB[3] * cross_AC_AD[3]

    # Volume = (1/6) * |triple_product|
    volume = abs(triple_product) / 6.0
    return volume
end # Mesh element volume

function areaTriangle(xt,yt,zt)
    Atr = 0.5*sqrt(det([xt';yt';[1 1 1]])^2 + det([yt';zt';[1 1 1]])^2 + det([zt';xt';[1 1 1]])^2);
    return Atr
end # Area of the 3D triangle

# Holds the mesh information needed for FEM simulations
mutable struct MESH
    # Node coordinates
    p::Matrix{Float64}

    # Connectivity list
    t::Matrix{Int32}
    
    # Surface triangles
    surfaceT::Matrix{Int32}
    
    # Elements inside material
    InsideElements::Vector{Int32}
    
    # Nodes inside material
    InsideNodes::Vector{Int32}
    
    # Volume of each mesh element
    VE::Vector{Float64}
    
    # Normal of each surface triangle
    normal::Matrix{Float64}
    
    nv::Int32       # Number of nodes
    nt::Int32       # Number of elements
    ne::Int32       # Number of surface elements
    nInside::Int32  # Number of elements for the inside cells

    # Constructor
    MESH() = new()
end

