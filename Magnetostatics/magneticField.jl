#=
    Simulation:
        Simulates the magnetostatic interaction between a magnetic plate
        and a uniform external magnetic field
        The plate has a uniform, constant magnetic permeability
    
    Output:
        Expect a 3D tetrahedral mesh as a MESH() struct and two figure. One for the mesh
        and one for the internal magnetic field of the plate. 

    Note:
        You can replace the local stiffness matrix with a C++ version by adding
        the FEMc.cpp file to the directory and replacing the function call by the
        C++ wrapper function
=#

# Gmsh powers the mesh generation and volume handling
using Gmsh
include("../gmsh_wrapper.jl")  # FEMjl functions to operate Gmsh and
                            # output the mesh in a more familiar format

# Required for solving matrix equations and exploit the sparsity of the matrices
using LinearAlgebra, SparseArrays

# For plots
using GLMakie

# Include FEM functions
include("../FEMjl.jl")

# View the mesh processed by FEMjl, generated with gmsh, using Makie
function viewMesh(mesh)
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

    wait(display(fig))

    # Save figure
    if saveFigure
        save("H.png",fig)
    end

end # Scatter plot of magnetic field


function main(meshSize=0,localSize=0,showGmsh=true,saveMesh=false)
    #=
        Makes a model with cubes and spheres and refines the mesh on the spheres
    
        Input:
            meshSize  - Mesh size (0 = let gmsh choose)
            localSize - Size of mesh in every volume beyond the container (0 for no local refinement)
            saveMesh  - Save mesh to a FEMCE compatible format 

    =#
    
    # Applied field
    mu0 = pi*4e-7      # vacuum magnetic permeability
    Hext = [1,0,0]     # T
    
    # Relative magnetic permeability
    permeability::Float64 = 1+2

    # Create a geometry
    gmsh.initialize()

    # >> Model
    # Create an empty container
    # container = addCuboid([0,0,0],[5,5,5])
    container = addSphere([0,0,0],5)

    # Get how many surfaces compose the bounding shell
    temp = gmsh.model.getEntities(2)            # Get all surfaces of current model
    bounding_shell_n_surfaces = 1:length(temp)    # Get the number of surfaces in the bounding shell


    # List of cells inside the container
    cells = []

    # Add another object inside the container
    addSphere([0,0,0],1,cells,true)
    # addCuboid([0,0,0],[1.65,1.65,0.04],cells,true)


    # Fragment to make a unified geometry
    _, fragments = gmsh.model.occ.fragment([(3, container)], cells)
    gmsh.model.occ.synchronize()

    # Update container volume ID
    container = fragments[1][1][2]

    # Generate Mesh
    mesh = Mesh(cells,meshSize,localSize,saveMesh)
    
    # Get bounding shell surface id
    shell_id = gmsh.model.getAdjacencies(3, container)[2]

    # Must remove the surface Id of the interior surfaces
    shell_id = shell_id[bounding_shell_n_surfaces] # All other, are interior surfaces

    if showGmsh
        gmsh.fltk.run()
    end
    gmsh.finalize()

    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))
    println("Bounding shell: ",shell_id)
    
    # View the mesh using Julia instead of Gmsh
    viewMesh(mesh)

    # FEM

    # Relative magnetic permeability 
    mu::Vector{Float64} = ones(mesh.nt);
    mu[mesh.InsideElements] .= permeability

    # Boundary conditions
    RHS = BoundaryIntegral(mesh,Hext,shell_id)

    # Lagrange multiplier technique
    Lag = lagrange(mesh)

    # Stiffness matrix
    A = @time stiffnessMatrix(mesh,mu) 
    
    # Extend the matrix for the Lagrange multiplier technique
    mat = [A Lag;Lag' 0]

    # Magnetic scalar potential
    u = mat\[-RHS;0]
    u = u[1:mesh.nv]

    #= Example of calling C++ to calculate the local stiffness matrix
        
        # Julia local stiffness matrix
        Ak::Matrix{Float64} = @time localStiffnessMatrix(mesh,mu)

        # C++ Local stiffness matrix
        @. mesh.t -= 1 # C++ index starts at 0
        Ak::Matrix{Float64} = @time CstiffnessMatrix(mesh.p,mesh.t,mesh.VE,mu)
    
    =#

    # Magnetic field | Normalized by mu0 -> H_true = H/mu0
    H_vectorField::Matrix{Float64} = zeros(mesh.nt,3)
    for k in 1:mesh.nt
        nds = mesh.t[:,k];

        # Sum the contributions
        for nd in nds
            # obtain the element parameters
            _,b,c,d = abcd(mesh.p,nds,nd)

            H_vectorField[k,1] = H_vectorField[k,1] - u[nd]*b;
            H_vectorField[k,2] = H_vectorField[k,2] - u[nd]*c;
            H_vectorField[k,3] = H_vectorField[k,3] - u[nd]*d;
        end
    end

    # Magnetic field intensity
    H::Vector{Float64} = zeros(mesh.nt)
    for k in 1:mesh.nt
        H[k] = sqrt(H_vectorField[k,1]^2+H_vectorField[k,2]^2+H_vectorField[k,3]^2)
    end

    # Magnetization
    chi::Vector{Float64} = mu .- 1;
    M_vectorField::Matrix{Float64} = zeros(mesh.nInside,3)
    M::Vector{Float64} = zeros(mesh.nInside)
    for ik in 1:mesh.nInside
        k = mesh.InsideElements[ik]
        
        M_vectorField[ik,1] = chi[k]*H_vectorField[k,1]
        M_vectorField[ik,2] = chi[k]*H_vectorField[k,2]
        M_vectorField[ik,3] = chi[k]*H_vectorField[k,3]

        M[ik] = chi[k]*H[k]
    end

    # Element centroids
    centroids::Matrix{Float64} = zeros(mesh.nt,3)
    for k in 1:mesh.nt
        nds = mesh.t[:,k]
        centroids[k,1] = sum(mesh.p[1,nds])/4
        centroids[k,2] = sum(mesh.p[2,nds])/4
        centroids[k,3] = sum(mesh.p[3,nds])/4
    end

    # Plot result | Uncomment "using GLMakie"
    plotHField(mesh,centroids,H,true)

    # Analytic result
    S::Vector{Float64} = zeros(mesh.nInside)
    dif::Float64 = 0.0  # Average difference
    v::Float64 = 0      # Total mesh volume of magnetic region
    for ik in 1:mesh.nInside
        k = mesh.InsideElements[ik]
        v += mesh.VE[k]

        S[ik] = (3*chi[k]/(3+chi[k]))*norm(Hext)
        
        dif += 100*abs(S[ik]-M[ik])/S[ik] *mesh.VE[k]
    end

    println("100*|M - Manalytic|/M_an.. = ",dif/v," %")


end # end of main

meshSize = 10
localSize = 0.1
showGmsh = false
saveMesh = false

main(meshSize,localSize,showGmsh,saveMesh)
