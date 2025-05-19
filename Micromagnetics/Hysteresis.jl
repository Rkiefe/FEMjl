#=
    Micromagnetic simulation to obtain hysteresis curves
=#

# ------------------------------------------
# Gmsh powers the mesh generation and volume handling
include("../gmsh_wrapper.jl")   # FEMjl functions to operate Gmsh and
                                # output the mesh in a more familiar format

# Include FEM functions
include("../FEMjl.jl")

# For plots
using GLMakie
# ------------------------------------------

include("SteepestDescent.jl")

# ------------------------------------------
function main()
    meshSize::Float64 = 1000
    localSize::Float64 = 5

    # Constants
    mu0::Float64 = pi*4e-7          # vacuum magnetic permeability
    giro::Float64 = 2.210173e5 /mu0 # Gyromagnetic ratio (rad T-1 s-1)
    dt::Float64 = 0.028/giro        # Time step in reduced units (seconds per gyro)
    
    maxTorque::Float64 = 1e-9       # Stop criteria of the relax function

    damp::Float64 = 1             # Damping parameter (dimensionless [0,1])
    precession::Bool = false         # Include precession or not

    # Dimension of the magnetic material (rectangle)
    L::Vector{Float64} = [512,128,30]
    scl::Float64 = 1e-9                 # scale of the geometry | (m -> nm)

    # Conditions
    Ms::Float64   = 800e3               # Magnetic saturation (A/m)
    Aexc::Float64 = 13e-12              # Exchange   (J/m)
    Aan::Float64  = 0.0                 # Anisotropy (J/m3)
    uan::Vector{Float64}  = [1,0,0]     # easy axis direction
    # The applied field is no longer constant
    
    
    # Convergence criteria
    maxAtt::Int32 = 4_000         # max number of iterations for the LL solver

    # Create a geometry
    # ------------------------------------------
    gmsh.initialize()

    # >> Model
    # Create an empty container
    container = addSphere([0,0,0],5*maximum(L))
    cells = [] # List of cells inside the container

    # Get how many surfaces compose the bounding shell
    temp = gmsh.model.getEntities(2)                # Get all surfaces of current model
    bounding_shell_n_surfaces = 1:length(temp)      # Get the number of surfaces in the bounding shell

    # Add another object inside the container
    addCuboid([0,0,0],L,cells,true)

    # Fragment to make a unified geometry
    _, fragments = gmsh.model.occ.fragment([(3, container)], cells)
    gmsh.model.occ.synchronize()

    # Update container volume ID
    container = fragments[1][1][2]

    # Generate Mesh
    mesh = Mesh(cells,meshSize,localSize,false)
    
    # Get bounding shell surface id
    mesh.shell_id = gmsh.model.getAdjacencies(3, container)[2]

    # Must remove the surface Id of the interior surfaces
    mesh.shell_id = mesh.shell_id[bounding_shell_n_surfaces] # All other, are interior surfaces

    # Finalize Gmsh and show mesh properties
    gmsh.finalize()
    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))
    # ------------------------------------------

    # Volume of elements of each mesh node | Needed for the demagnetizing field
    Vn::Vector{Float64} = zeros(mesh.nv)

    # Integral of basis function over the domain | Needed for the exchange field
    nodeVolume::Vector{Float64} = zeros(mesh.nv)
    
    for ik in 1:mesh.nInside
        k = mesh.InsideElements[ik]
        Vn[mesh.t[:,k]]         .+= mesh.VE[k]
        nodeVolume[mesh.t[:,k]] .+= mesh.VE[k]/4
    end
    Vn = Vn[mesh.InsideNodes]
    nodeVolume = nodeVolume[mesh.InsideNodes]

    # Stiffness matrix | for the demagnetizing field
    AD = stiffnessMatrix(mesh)

    # Stiffness matrix, with only the internal mesh elements | for the exchange field
    Ak::Matrix{Float64} = zeros(4*4,mesh.nt)
    A = spzeros(mesh.nv,mesh.nv)
    begin # Make a local scope to keep the workspace clean
        
        b::Vector{Float64} = zeros(4)
        c::Vector{Float64} = zeros(4)
        d::Vector{Float64} = zeros(4)
        aux::Matrix{Float64} = zeros(4,4)
            
        # Only go through the magnetic elements
        for ik in 1:mesh.nInside
            k = mesh.InsideElements[ik]
            for i in 1:4
                _,b[i],c[i],d[i] = abcd(mesh.p,mesh.t[:,k],mesh.t[i,k])
            end
            aux = mesh.VE[k]*(b*b' + c*c' + d*d')
            Ak[:,k] = aux[:] # vec(aux)
        end

        # Update sparse global matrix
        n = 0
        for i in 1:4
            for j in 1:4
                n += 1
                A += sparse(Int.(mesh.t[i,:]),Int.(mesh.t[j,:]),Ak[n,:],mesh.nv,mesh.nv)
            end
        end

        # Remove all exterior nodes
        A = A[mesh.InsideNodes,mesh.InsideNodes]
    end # Local scope to calculate the stiffness matrix of the exchange field

    # Dirichlet boundary condition
    fixed::Vector{Int32} = findNodes(mesh,"face",mesh.shell_id)
    free::Vector{Int32} = setdiff(1:mesh.nv,fixed)
    # ------------------------------------------

    # Initial magnetization - random
    m::Matrix{Float64} = zeros(3,mesh.nv)
    begin # Set the initial magnetization
        theta::Vector{Float64} = 2*pi*rand(mesh.nInsideNodes)
        phi::Vector{Float64} = pi*rand(mesh.nInsideNodes)
        for i = 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            m[:,i] = [sin(phi[i])*cos(theta[i]),sin(phi[i])*sin(theta[i]),cos(phi[i])]
            
            # Normalize the magnetization
            # m[:,i] ./= norm(m[:,i])
        end
    end
    
    # ----------------------------------------------------------------

    Hext::Vector{Float64} = vcat(0:1e-3:0.1,0.1:-1e-3:-0.1,-0.1:1e-3:0.1)./mu0 # A/m

    # Average magnetization over different applied fields
    M_H::Matrix{Float64} = zeros(3,length(Hext))

    # Start with Hap = [0,0,0]
    Hap::Vector{Float64}  = [0.0,0.0,0.0] # A/m

    # Minimize the energy by the dynamic Landau-Lifshitz equation
    m, Heff, _, M_avg = relax(mesh,scl,m,Ms,Aexc,Aan,uan,Hap,dt,maxTorque,giro,damp,precession,maxAtt)
    
    # Hysteresis curve
    for ih in 1:length(Hext)
        Hap[1] = Hext[ih]

        # Steepest descent energy minimization
        m, Heff, M_avg = steepestDescent(mesh,scl,m,Ms,Aexc,Aan,uan,Hap,maxTorque,giro,maxAtt)
        M_H[:,ih] = M_avg[:,end]
    end

    fig = Figure()
    ax = Axis(fig[1,1])
    
    i::Int32 = 1
    j::Int32 = (0.1-0)/1e-3 + 1
    scatter!(ax, Hext[i:j], M_H[1,i:j]) # , label = "M_x"
    
    i = j + 1
    j += (0.1+0.1)/1e-3 + 1
    scatter!(ax, Hext[i:j], M_H[1,i:j])

    i = j + 1
    j += (0.1+0.1)/1e-3 + 1
    scatter!(ax, Hext[i:j], M_H[1,i:j])

    # scatter!(ax, Hext./mu0, M_H[1,:], label = "M_x")
    # scatter!(ax, Hext./mu0, M_H[2,:], label = "M_y")
    # scatter!(ax, Hext./mu0, M_H[3,:], label = "M_z")
    # axislegend() # position = :rt

    save("M_H.png",fig)
    wait(display(fig))
end

main()