#=
    3D Finite element implementation of the Landau-Lifshitz equation with a semi-implciti
    time step, based on https://doi.org/10.1109/TMAG.2008.2001666 (Oriano 2008)
=#

# ------------------------------------------
# Gmsh powers the mesh generation and volume handling
include("../gmsh_wrapper.jl")  # FEMjl functions to operate Gmsh and
                            # output the mesh in a more familiar format

# Include FEM functions
include("../FEMjl.jl")

# For plots
using GLMakie
# ------------------------------------------

# Find new magnetization after time iteration
function timeStep(m,H,Hold,Heff,dt,giro,damp::Float64=1,precession::Bool=true)
    #=
        Repeats the search of a new magnetization until the solution is stable
    =#

    d = dt*giro/2

    # The new magnetization after the time step
    m2::Vector = zeros(3)

    # Initial guess of the new magnetic field
    H12 = 3/2 *H - 0.5 *Hold

    # Repeat m12 = H12(m12) until m12 doesnt change
    aux::Vector{Float64} = zeros(3)
    att::Int32 = 0
    err::Float64 = 1.0
    while err > 1e-6
        att += 1
        mat::Matrix{Float64} = [1 d*H12[3] -d*H12[2];
                                -d*H12[3] 1 d*H12[1];
                                d*H12[2] -d*H12[1] 1]
        m2 = mat\(m - d*cross(m,H12))

        # New m12 from H12
        m12 = 0.5*(m + m2)

        # New H12
        if precession
            H12 = Heff + damp*cross(m12,Heff)
        else
            H12 = damp*cross(m12,Heff)
        end

        # Max difference between m2 and previous m2
        err = maximum(abs.(m2-aux))
        # println(err)

        aux = m2
        if att > 100
            println("Time step did not converge in ",att," steps")
            break
        end
    end

    return m2
end # Find new magnetization after time iteration

# ------------------------------------------
function main()
    meshSize::Float64 = 200
    localSize::Float64 = 5

    # Constants
    mu0::Float64 = pi*4e-7          # vacuum magnetic permeability
    giro::Float64 = 2.210173e5 /mu0 # Gyromagnetic ratio (rad T-1 s-1)
    dt::Float64 = 0.028/giro        # Time step in reduced units (seconds per gyro)
    totalTime::Float64 = 0.4        # Total time of spin dynamics simulation (ns)
    damp::Float64 = 0.1             # Damping parameter (dimensionless [0,1])
    precession::Bool = true         # Include precession or not

    # Dimension of the magnetic material (rectangle)
    L::Vector{Float64} = [100,100,5]
    scl::Float64 = 1e-9                 # scale of the geometry | (m -> nm)
    
    # Conditions
    Ms::Float64   = 860e3              # Magnetic saturation (A/m)
    Aexc::Float64 = 13e-12             # Exchange   (J/m)
    Aan::Float64  = 0.0                # Anisotropy (J/m3)
    uan::Vector{Float64}  = [1,0,0]    # easy axis direction
    Hap::Vector{Float64}  = [0,50e3,0] # A/m

    # Convergence criteria
    maxDeviation::Float64 = 1e-5     # Maximum difference between current and previous <M>
    maxAtt::Int32 = Int(1e4)         # max number of iterations for the LL solver

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
    shell_id = gmsh.model.getAdjacencies(3, container)[2]

    # Must remove the surface Id of the interior surfaces
    shell_id = shell_id[bounding_shell_n_surfaces] # All other, are interior surfaces

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
    fixed::Vector{Int32} = findNodes(mesh,"face",shell_id)
    free::Vector{Int32} = setdiff(1:mesh.nv,fixed)
    # ------------------------------------------

    # Magnetization field
    m::Matrix{Float64} = zeros(3,mesh.nv)
    for i = 1:mesh.nInsideNodes
        nd = mesh.InsideNodes[i]
        m[1,i] = 1
    end

    # Magnetic fields
    Heff::Matrix{Float64} = zeros(3,mesh.nInsideNodes) .+ mu0*Hap
    
    # Demagnetizing field
    Hd::Matrix{Float64} = demagField(mesh,fixed,free,AD,m)

    # Exchange field (T)
    Hexc::Matrix{Float64} = -2*Aexc.* (A*m[:,mesh.InsideNodes]')'

    # Anisotropy field (T)
    Han::Matrix{Float64} = zeros(3,mesh.nInsideNodes)
    for i in 1:mesh.nInsideNodes
        nd = mesh.InsideNodes[i]
        Han[:,i] = 2*Aan/Ms*dot(m[:,nd],uan).*uan
    end
    
    # Convert to proper unis
    @simd for i in 1:3
        Hd[i,:]     .*= mu0*Ms./Vn
        Hexc[i,:]   ./= Ms*scl^2 .*nodeVolume
    end

    # Effective field
    Heff += Hd + Hexc + Han

    # H = Heff + damp* m cross Heff
    H::Matrix{Float64} = zeros(3,mesh.nInsideNodes)
    for i in 1:mesh.nInsideNodes
        nd = mesh.InsideNodes[i]
        H[:,i] = Heff[:,i] + damp.*cross(m[:,nd],Heff[:,i])
    end

    # Time step
    # ------------------------------------------

    # Average magnetization over time
    M_avg::Matrix{Float64} = zeros(3,maxAtt)

    # Energy density
    E::Float64    = 0
    Eext::Float64 = 0
    Ed::Float64   = 0
    Eexc::Float64 = 0
    Ean::Float64  = 0

    E_time::Vector{Float64} = zeros(maxAtt)
    torque_time::Vector{Float64} = zeros(maxAtt)

    Hold::Matrix{Float64} = deepcopy(H)
    
    t::Float64 = 0  # Time (s)
    it::Int32 = 0   # Iteration step
    while 1e9*t < totalTime && it < maxAtt
        t += dt
        it += 1
        println(1e9 * t)

        # New magnetization
        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            m[:,nd] = timeStep(m[:,nd],H[:,i],Hold[:,i],Heff[:,i],dt,giro,damp,precession)
        end

        # New magnetic field
        Heff = zeros(3,mesh.nInsideNodes) .+ mu0*Hap
        
        # Demagnetizing field
        Hd = demagField(mesh,fixed,free,AD,m)

        # Exchange field (T)
        Hexc = -2*Aexc.* (A*m[:,mesh.InsideNodes]')'

        # Anisotropy field (T)
        Han = zeros(3,mesh.nInsideNodes)
        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            Han[:,i] = 2*Aan/Ms*dot(m[:,nd],uan).*uan
        end
        
        # Convert to proper unis
        @simd for i in 1:3
            Hd[i,:]     .*= mu0*Ms./Vn
            Hexc[i,:]   ./= Ms*scl^2 .*nodeVolume
        end

        # Effective field
        Heff += Hd + Hexc + Han

        # H = Heff + damp* m cross Heff
        H = zeros(3,mesh.nInsideNodes)
        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            H[:,i] = Heff[:,i] + damp.*cross(m[:,nd],Heff[:,i])
        end

        # Store last magnetic field
        Hold = deepcopy(H)

        # Energy density
        Eext = 0
        Ed   = 0
        Eexc = 0
        Ean  = 0

        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            Eext -= dot(m[:,nd],mu0*Hap)
            Ed   -= 0.5*dot(m[:,nd],Hd[:,i])
            Eexc -= 0.5*dot(m[:,nd],Hexc[:,i])
            Ean  -= 0.5*dot(m[:,nd],Han[:,i])
        end
        E = mu0*Ms*(Eext + Ed + Eexc + Ean)
        E_time[it] = E # Store the energy for each time step

        # Average torque
        dtau::Float64 = 0
        for i = 1:mesh.nInsideNodes
            dtau += norm(cross(m[:,mesh.InsideNodes[i]],Heff[:,i]))
        end
        torque_time[it] = dtau/mesh.nInsideNodes
        
        # Average magnetization
        M_avg[:,it] = mean(m[:,mesh.InsideNodes],2)

    end # End of time iteration

    # Remove excess zeros
    M_avg       = M_avg[:,1:it]
    E_time      = E_time[1:it]
    torque_time = torque_time[1:it]

    time::Vector{Float64} = 1e9*dt .*(1:it)

    fig = Figure()
    ax = Axis(  fig[1,1], 
                xlabel = "Time (ns)", 
                ylabel = "<M> (kA/m)",
                title = "Micromagnetic simulation")

    scatter!(ax,time,Ms/1000 .*M_avg[1,:], label = "M_x")
    scatter!(ax,time,Ms/1000 .*M_avg[2,:], label = "M_y")
    scatter!(ax,time,Ms/1000 .*M_avg[3,:], label = "M_z")
    axislegend() # position = :rt

    wait(display(fig))
end

main()