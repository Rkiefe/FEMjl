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

# Gmsh powers the mesh generation and volume handling
using Gmsh

# Local mesh refinement on target cell
function refineCell(cell,localSize,meshSize)
    #=
        Sets every volume in 'cell' to be locally refined with target 'localSize' 
    =#

    if isempty(cell)
        println("No cell to refine, function returned")
        return
    end

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
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", meshSize)   # Full refinement up to this distance

    # Adds a smoother element size transition (if set to 1)
    gmsh.model.mesh.field.setNumber(threshold_field, "Sigmoid", 0) # 0 -> linear; 1 -> sigmoid

    # Set this if you don't want to preserve the boundary shape
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0) 


    # Set the background mesh field
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    # -- Use the minimum of all the fields as the background mesh field --
    # min_field = gmsh.model.mesh.field.add("Min")
    # gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    # gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

end # Local mesh refinement on target cell

# Make a cuboid based on its center
function addCuboid(position,dimensions,cells=[],updateCells=false)
    #=
        Makes a cuboid based on its centroid position
        Updates the cells list if this cuboid is not meant to be
        the container of the simulation
    =#

    r = position - dimensions/2
    box = gmsh.model.occ.addBox(r[1], r[2], r[3], dimensions[1], dimensions[2], dimensions[3])
    
    if updateCells # default is false
        cells = append!(cells,[(3,box)])
    end

    # Sync kernel before exiting
    gmsh.model.occ.synchronize()
    return box
end # Make a cuboid based on its center

# Make a sphere
function addSphere(position,radius,cells=[],updateCells=false)
    #=
        Inputs:
            Position vector
            radius value
            cells <- a list of volumes (cells) that are inside a container
    =# 
    
    # Add a sphere to the current model
    sphere = gmsh.model.occ.addSphere(position[1],position[2],position[3],radius)

    # If sphere is not the container
    if updateCells # default is false 
        cells = append!(cells,[(3,sphere)])
    end

    # Sync kernel before exiting
    gmsh.model.occ.synchronize()
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
    dimensions = scale*L

    box = addSphere(center,dimensions)
    # box = addCuboid(center,dimensions)

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

function findNodes(mesh,region::String,id)
    # region    - face | volume
    # id        - Int or vector of Int 

    nodesFound::Vector{Int32} = zeros(mesh.nv)
    n::Int32 = 0 # Number of nodes found
    
    if region == "face" || region == "Face" # added "Face" to handle variations

        # Go to each surface triangle
        for s in 1:mesh.ne
            # Get the surface id of current triangle
            current_Id::Int32 = mesh.surfaceT[end,s]
            if current_Id in id
                # Nodes of current triangle
                nds = @view mesh.surfaceT[1:3,s]

                # Update number of nodes found
                for nd in nds
                    if nodesFound[nd] < 1   # Only count those who were not found yet
                        n += 1                  # count it
                    end
                end

                # Update the nodes found with desired face ID
                nodesFound[nds] .= 1
            end
        end

        # Prepare the output
        nodes::Vector{Int32} = zeros(n)
        j::Int32 = 0
        for nd in 1:mesh.nv
            if nodesFound[nd] > 0
                j += 1
                nodes[j] = nd
            end
        end

    else # volume
        # not implemented yet
    end

    return nodes
end

# Save matrix to .txt file
function save2file(fileName,input)
    # Saves matrix to a .txt file
    open(fileName, "w") do io
        for row in eachrow(input)
            println(io, join(row, " , "))  # Space-separated
        end
    end
end # Save matrix to .txt file

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
    if !isempty(cells)
        for i in cells
            append!(volumeID,i[2])
        end
    end

    # >> Mesh settings
    
    # Set local mesh size
    if localSize>0 && !isempty(cells)
        refineCell(cells,localSize,meshSize) # Set local refinement on the sphere Cell
    end

    # Set maximum element size
    if meshSize > 0
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshSize)
    end

    # -- Generate mesh --
    # gmsh.option.setNumber("Mesh.Algorithm",6)
    gmsh.model.mesh.generate(3)

    # -- Optimize the mesh --
        # "" (default), "NetGen", "HighOrder", 
        # "Relocate3D", "HighOrderElastic", "UntangleMeshGeometry"
    gmsh.model.mesh.optimize("UntangleMeshGeometry")

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
    if !isempty(cells)
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
    else
        mesh.InsideElements = []
        mesh.nInside = 0
    end

    # Inside nodes
    aux::Matrix{Int32} = mesh.t[:,mesh.InsideElements]
    mesh.InsideNodes = unique(vec(aux))
    mesh.nInsideNodes = length(mesh.InsideNodes)

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
    
    nv::Int32           # Number of nodes
    nt::Int32           # Number of elements
    ne::Int32           # Number of surface elements
    nInside::Int32      # Number of elements for the inside cells
    nInsideNodes::Int32 # Numberf of nodes for the inside cells
    
    
    shell_id # Id of the surface boundaries of the bounding shell

    # Constructor
    MESH() = new()
end

# View the mesh using Makie
function viewMesh(mesh::MESH)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=:data, title="")
    
    # Convert surface triangles to Makie format
    faces = [GLMakie.GLTriangleFace(mesh.surfaceT[1,i], 
                                    mesh.surfaceT[2,i], 
                                    mesh.surfaceT[3,i]) for i in 1:size(mesh.surfaceT,2)]
    
    # Create mesh plot using surface triangles
    mesh!(ax, mesh.p', faces,
            color=:lightblue,
            transparency=true,
            alpha=0.3)

    wait(display(fig))
end # View the mesh using Makie
