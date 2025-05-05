#=
        Author: Rodrigo Kiefe

    Summary:
    A finite element implementation in Julia, demonstrating the calculation of
    the stiffness matrix and how it can be done with C++ instead, called within
    the Julia environment.

    Description:
    This repository aims to be a foundation. Something users can build on, to create 
    ground up implementations of the finite element method without compromising performance.
    This code base also demonstrates interoperability with C++ and high performance packages
    such as Eigen. As an example, this code shows how to calculate the dense, local stiffness matrix
    in C++, called from the Julia environment, with significant performance gains.
    Two examples are available: 1) Task intensive, called once.
                                2) Low compute time, called millions of times.
    In both scenarios, Julia benefits from calling the C++ variants of the same functions.
=#

# For the solver
using Gmsh, LinearAlgebra, SparseArrays
include("gmsh_wrapper.jl")

# For plots | Uncomment the plot section of "main()"
using GLMakie

# View the mesh processed by FEMjl, generated with gmsh, using Makie
function viewMesh(mesh)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect=:equal, title="")
    
    # Convert surface triangles to Makie format
    faces = [GLMakie.GLTriangleFace(mesh.surfaceT[1,i], 
                                    mesh.surfaceT[2,i], 
                                    mesh.surfaceT[3,i]) for i in 1:size(mesh.surfaceT,2)]
    
    # Create mesh plot using surface triangles
    mesh!(ax, mesh.p', faces,
            color=:lightblue,
            transparency=true,
            alpha=0.3)

    screen = GLMakie.Screen()
    display(screen,fig)
    while isopen(screen)
        sleep(0.1)
    end

end # View the mesh using Makie

# Scatter plot of magnetic field
function plotHField(mesh,centroids::Matrix{Float64},H::Vector{Float64},saveFigure)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect = :data, title="Magnetic field H")

    # Add |H|
    scatterPlot = scatter!(ax, 
        centroids[mesh.InsideElements,1],
        centroids[mesh.InsideElements,2],
        centroids[mesh.InsideElements,3], 
        color = H[mesh.InsideElements], 
        colormap=:rainbow, 
        markersize=20 .* mesh.VE[mesh.InsideElements]./maximum(mesh.VE[mesh.InsideElements]))

    # Add colorbar
    Colorbar(fig[1, 2], scatterPlot, label="H field strength")

    screen = GLMakie.Screen()
    display(screen,fig)
    while isopen(screen)
        sleep(0.1)
    end

    # Save figure
    if saveFigure
        save("H.png",fig)
    end
end # Scatter plot of magnetic field

# FEM linear basis function
function abcd(p::Matrix{Float64},nodes::Vector{Int32},nd::Int32)
    n1,n2,n3 = nodes[nodes .!= nd]
    x = @view p[1, [nd, n1, n2, n3]]
    y = @view p[2, [nd, n1, n2, n3]]
    z = @view p[3, [nd, n1, n2, n3]]

    M::Matrix{Float64} = [1.0 x[1] y[1] z[1];
                         1.0 x[2] y[2] z[2];
                         1.0 x[3] y[3] z[3];
                         1.0 x[4] y[4] z[4]]

    r::Vector{Float64} = M\[1;0;0;0]

    return r[1],r[2],r[3],r[4] # f= a + bx + cy + dz
end # Basis function coef.

# Sparse, global stiffness matrix
function stiffnessMatrix(mesh,f::Vector{Float64})
    A = spzeros(mesh.nv,mesh.nv)

    # Local stiffness matrix
    Ak::Matrix{Float64} = localStiffnessMatrix(mesh,f)

    # Update sparse global matrix
    n = 0
    for i in 1:4
        for j in 1:4
            n += 1
            A += sparse(Int.(mesh.t[i,:]),Int.(mesh.t[j,:]),Ak[n,:],mesh.nv,mesh.nv)
        end
    end

    return A
end # Sparse, global stiffness matrix

# Lagrange multiplier technique
function lagrange(mesh)
    C = zeros(mesh.nv,1);
    for k in 1:mesh.nt
        nds::AbstractVector{Int32} = @view mesh.t[:,k];   # Nodes of that element

        # For each node of that element
        for i in 1:length(nds)
            C[nds[i]] = C[nds[i]] + mesh.VE[k]/4;
        end
    end
    return C
end # Lagrange multiplier technique

# Boundary condition
function BoundaryIntegral(mesh,F,shell_id)
    RHS = zeros(mesh.nv,1);
    for s in 1:mesh.ne

        # Only integrate over the desired elements
        if !(mesh.surfaceT[4,s] in shell_id)
            continue
        end

        # Nodes of the surface triangle
        nds = @view mesh.surfaceT[1:3,s];
        
        # Area of surface triangle
        areaT = areaTriangle(mesh.p[1,nds],mesh.p[2,nds],mesh.p[3,nds]);
        
        RHS[nds] .+= dot(mesh.normal[:,s],F)*areaT/3;
    end

    return RHS
end # Boundary conditions

# Wrapper for C++ function to get local stiffness matrix 
function CstiffnessMatrix(p::Matrix{Float64}, t::Matrix{Int32}, VE::Vector{Float64}, mu::Vector{Float64})
    # Get the matrix dimensions
    nt::Int32 = size(t,2)
    nv::Int32 = size(p,2)

    # Ready the output
    Ak::Matrix{Float64} = zeros(16,nt)
    
    # Call C++ function
    @ccall "FEMc.so".stiffnessMatrix(Ak::Ptr{Float64}, p::Ptr{Float64}, t::Ptr{Int32}, nv::Int32, nt::Int32, VE::Ptr{Float64}, mu::Ptr{Float64})::Cvoid

    return Ak
end # Wrapper for C++ function to get local stiffness matrix 

# Local stiffnessmatrix in 100% Julia
function localStiffnessMatrix(mesh,f::Vector{Float64})
    Ak::Matrix{Float64} = zeros(4*4,mesh.nt)
    b::Vector{Float64} = zeros(4)
    c::Vector{Float64} = zeros(4)
    d::Vector{Float64} = zeros(4)
    aux::Matrix{Float64} = zeros(4,4)
    
    for k in 1:mesh.nt
        for i in 1:4
            _,b[i],c[i],d[i] = abcd(mesh.p,mesh.t[:,k],mesh.t[i,k])
        end
        aux = mesh.VE[k]*f[k]*(b*b' + c*c' + d*d')
        Ak[:,k] = aux[:] # vec(aux)
    end

    return Ak
end # Local stiffnessmatrix in 100% Julia


function main(meshSize=0,localSize=0,showGmsh=true,saveMesh=false)
    #=
        An example of creating your own model with simple shapes
    =#
    
    # Create a geometry
    gmsh.initialize()

    # >> Model
    # Create an empty container
    box = addCuboid([0,0,0],[4,4,4])

    # List of cells inside the container
    cells = []

    # Add another object inside the container
    
    # addSphere([0,0,0],0.5,cells)
    addCuboid([0,0,0],[1.65,1.65,0.04],cells,true)

    # Fragment to make a unified geometry
    _, fragments = gmsh.model.occ.fragment([(3, box)], cells)
    gmsh.model.occ.synchronize()

    # Update container volume ID
    box = fragments[1][1][2]

    # Generate Mesh
    mesh = Mesh(cells,meshSize,localSize,saveMesh)
    
    # Get bounding shell surface id
    shell_id = gmsh.model.getAdjacencies(3, box)[2]

    # Must remove the surface Id of the interior surfaces
    shell_id = shell_id[1:6] # All other, are interior surfaces

    if showGmsh
        gmsh.fltk.run()
    end
    gmsh.finalize()

    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))

    # View the mesh using Julia instead of Gmsh
    viewMesh(mesh)

end # end of main

function geometryFromCAD(meshSize=0,localSize=0,showGmsh=false,saveMesh=false)
    
    # Create a geometry
    gmsh.initialize()
    
    # List of cells inside the container
    cells = []

    # >> Model
    box = importCAD("STEP_Models/cube.step",cells)

    # Fragment to make a unified geometry
    _, fragments = gmsh.model.occ.fragment([(3, box)], cells)
    gmsh.model.occ.synchronize()

    # Update container volume ID
    box = fragments[1][1][2]

    # Generate Mesh
    mesh = Mesh(cells,meshSize,localSize,saveMesh)
    
    # Get bounding shell surface id
    shell_id = gmsh.model.getAdjacencies(3, box)[2]

    # Must remove the surface Id of the interior surfaces
    shell_id = shell_id[1:6] # All other, are interior surfaces

    if showGmsh
        gmsh.fltk.run()
    end
    gmsh.finalize()

    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))

    # View the mesh using Julia instead of Gmsh
    viewMesh(mesh)

end

meshSize = 10
localSize = 0.1
showGmsh = false
saveMesh = false

main(meshSize,localSize,showGmsh,saveMesh)
geometryFromCAD()