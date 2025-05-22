#=
    Minimizes the energy by steepest descent and by the 
    Landau-Lifshitz equation.
=#


# For plots
using GLMakie


include("SteepestDescent.jl") # For the steepest descent algorithm

function main()
    meshSize::Float64 = 2_500
    localSize::Float64 = 50

    # Constants
    mu0::Float64 = pi*4e-7          # vacuum magnetic permeability
    giro::Float64 = 2.210173e5 /mu0 # Gyromagnetic ratio (rad T-1 s-1)
    dt::Float64 = 0.028/giro        # Time step (s)
    damp::Float64 = 1.0             # Damping parameter (dimensionless [0,1])
    precession::Bool = false         # Include precession or not

    # Dimension of the magnetic material (rectangle)
    L::Vector{Float64} = [512,128,30]
    scl::Float64 = 1e-9                 # scale of the geometry | (m -> nm)
    
    # Conditions
    Ms::Float64   = 800e3               # Magnetic saturation (A/m)
    Aexc::Float64 = 13e-12              # Exchange   (J/m)
    Aan::Float64  = 0.0                 # Anisotropy (J/m3)
    uan::Vector{Float64}  = [1,0,0]     # easy axis direction
    Hap::Vector{Float64}  = [0,0,0]     # A/m

    # Convergence criteria | Only used when totalTime != Inf
    maxTorque::Float64 = 1e-9          # Maximum difference between current and previous <M>
    maxAtt::Int32 = 10_000             # Maximum number of iterations in the solver
    
    # -- Create a geometry --
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
        # return
    # -----------------------

    # Magnetization field
    m::Matrix{Float64} = zeros(3,mesh.nv)
    m[1,mesh.InsideNodes] .= 1
    
    # Random initial magnetization
    # begin
    #     theta::Vector{Float64} = 2*pi*rand(mesh.nInsideNodes)
    #     phi::Vector{Float64} = pi*rand(mesh.nInsideNodes)
    #     for i = 1:mesh.nInsideNodes
    #         nd = mesh.InsideNodes[i]
    #         m[:,i] = [sin(phi[i])*cos(theta[i]),sin(phi[i])*sin(theta[i]),cos(phi[i])]
    #     end
    # end
    
    # Minimize energy by L. L. equation with damp = 1
    m, Heff, _, 
    M_avg, E_time, torque_time = LandauLifshitz(
                    mesh, 
                    scl, 
                    m, 
                    Ms, 
                    Aexc, 
                    Aan, 
                    uan, 
                    Hap, 
                    dt, 
                    giro, 
                    damp, 
                    precession, 
                    maxTorque, 
                    maxAtt
                    )
    
    # Steepest descent energy minimization
    m, Heff, 
    M_avg, E_time, torque_time = steepestDescent(
                    mesh,
                    scl,
                    m,
                    Ms,
                    Aexc,
                    Aan,
                    uan,
                    Hap,
                    giro,
                    maxTorque,
                    maxAtt)
    
    # Energy
    fig = Figure()
    ax = Axis(  fig[1,1], 
                xlabel = "Time (ns)", 
                ylabel = "Energy",
                title = "")
    scatter!(ax,1:length(E_time),E_time) # E_time

    # Log of torque
    ax = Axis(  fig[1,2], 
                xlabel = "Time (ns)", 
                ylabel = "Log(Torque)",
                title = "")
    scatter!(ax,1:length(torque_time),log10.(torque_time)) # E_time

    # Magnetization
    ax = Axis(  fig[1,3], 
                xlabel = "Time (ns)", 
                ylabel = "<M> (kA/m)",
                title = "Average Magnetization")

    scatter!(ax,1:size(M_avg,2),Ms/1000 .*M_avg[1,:], label = "M_x")
    scatter!(ax,1:size(M_avg,2),Ms/1000 .*M_avg[2,:], label = "M_y")
    scatter!(ax,1:size(M_avg,2),Ms/1000 .*M_avg[3,:], label = "M_z")
    axislegend() # position = :rt

    wait(display(fig))
end

main()