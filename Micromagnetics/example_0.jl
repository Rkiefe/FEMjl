#=
    Solves the Landau-Lifshitz for a high stress-test example
        A sphere without an exchange field and no damping has an analytic solution,
        a sinosoidal behavior of the <M> over time.

    The result is heavily dependent on the mesh quality
    and the bounding shell size
=#


# For plots
using GLMakie

include("LandauLifshitz.jl")

function main()
    meshSize::Float64 = 2_500
    localSize::Float64 = 5

    # Constants
    mu0::Float64 = pi*4e-7          # vacuum magnetic permeability
    giro::Float64 = 2.210173e5 /mu0 # Gyromagnetic ratio (rad T-1 s-1)
    dt::Float64 = 1e-12             # Time step (s)
    totalTime::Float64 = 0.1        # Total time of spin dynamics simulation (ns)
    damp::Float64 = 0.0             # Damping parameter (dimensionless [0,1])
    precession::Bool = true         # Include precession or not

    # Dimension of the magnetic material 
    # L::Vector{Float64} = [100,100,5]  # (rectangle)
    scl::Float64 = 1e-9                 # scale of the geometry | (m -> nm)
    
    # Conditions
    Ms::Float64   = 1400e3              # Magnetic saturation (A/m)
    Aexc::Float64 = 0                   # Exchange   (J/m)
    Aan::Float64  = 500e3               # Anisotropy (J/m3)
    uan::Vector{Float64}  = [1,0,0]     # easy axis direction
    Hap::Vector{Float64}  = [0,400e3,0] # A/m

    # Convergence criteria | Only used when totalTime != Inf
    maxTorque::Float64 = 0        # Maximum difference between current and previous <M>
    maxAtt::Int32 = 10_000             # Maximum number of iterations in the solver
    
    # -- Create a geometry --
        gmsh.initialize()

        # >> Model
        # Create an empty container
        container = addSphere([0,0,0],5*50)

        cells = [] # List of cells inside the container

        # Get how many surfaces compose the bounding shell
        temp = gmsh.model.getEntities(2)                # Get all surfaces of current model
        bounding_shell_n_surfaces = 1:length(temp)      # Get the number of surfaces in the bounding shell

        # Add another object inside the container
        # addCuboid([0,0,0],L,cells,true)
        addSphere([0,0,0],50,cells,true)

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
        # gmsh.fltk.run()
        gmsh.finalize()
        
        # Or load mesh from matlab
        # mesh = MESH()
        # mesh.t = readdlm("mesh/t.txt",','); mesh.nt = size(mesh.t,2)
        # mesh.p = readdlm("mesh/p.txt",','); mesh.nv = size(mesh.p,2)
        # mesh.surfaceT = readdlm("mesh/surfaceT.txt",','); mesh.ne = size(mesh.surfaceT,2)
        # mesh.InsideElements = vec(readdlm("mesh/InsideElements.txt",',')); mesh.nInside = length(mesh.InsideElements)
        # mesh.InsideNodes = vec(readdlm("mesh/InsideNodes.txt",',')); mesh.nInsideNodes = length(mesh.InsideNodes)
        # mesh.VE = vec(readdlm("mesh/VE.txt",','))
        # shell_id = [1] 
    # -----------------------

    println("Number of elements ",size(mesh.t,2))
    println("Number of Inside elements ",length(mesh.InsideElements))
    println("Number of nodes ",size(mesh.p,2))
    println("Number of Inside nodes ",length(mesh.InsideNodes))
    println("Number of surface elements ",size(mesh.surfaceT,2))
    # println("Bounding shell: ",mesh.shell_id)

    # viewMesh(mesh)
    
    # Magnetization field
    m::Matrix{Float64} = zeros(3,mesh.nv)
    m[1,mesh.InsideNodes] .= 1
    
    m, Heff, time, M_avg, E_time, torque_time =
    LandauLifshitz( mesh, 
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
                    maxAtt, 
                    totalTime)
    

    fig = Figure()
    ax = Axis(  fig[1,1], 
                xlabel = "Time (ns)", 
                ylabel = "<M> (kA/m)",
                title = "Micromagnetic simulation")

    scatter!(ax,time,Ms/1000 .*M_avg[1,:], label = "M_x")
    axislegend()

    save("M_time_Sphere.png",fig)
    wait(display(fig))
end

main()