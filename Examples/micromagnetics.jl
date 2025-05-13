#=
    Micromagnetic simulation with the Landau-Lifshitz eq. based on this papers

    L.L. eq integrator: https://ieeexplore.ieee.org/document/4717321
    Steepest descent energy minimizer: https://doi.org/10.1063/1.4862839
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
function plotHField(mesh,H::Matrix{Float64},saveFigure=false)
    
    # Calculate the norm of H
    H_norm::Vector{Float64} = zeros(mesh.nInsideNodes)
    @simd for i in 1:mesh.nInsideNodes
        H_norm[i] = sqrt(sum(H[:,i].^2))
    end


    fig = Figure()
    ax = Axis3(fig[1, 1], aspect = :data, title="Demagnetizing field")
    scatterPlot = scatter!(ax, 
        mesh.p[1,mesh.InsideNodes],
        mesh.p[2,mesh.InsideNodes],
        mesh.p[3,mesh.InsideNodes], 
        color = H_norm, 
        colormap=:rainbow)
    Colorbar(fig[1, 2], scatterPlot, label="Field strength")

    wait(display(fig))

    # Save figure
    if saveFigure
        save("H.png",fig)
    end
end # Scatter plot of magnetic field

# Norm of vector
function norm(v)
    return sqrt(sum(v.^2))
end

# Normalize D by N field, where D is expected to be 2 or 3
function normalizeField(m::Matrix{Float64})
    @simd for i in 1:size(m,2)
        m[:,i] ./= norm(m[:,i])
    end
end # Normalize field

# Initialize magnetization field
function initialMagnetization(mesh,random=true,direction=[1,0,0])
    m::Matrix{Float64} = zeros(3,mesh.nv) 
    theta::Vector{Float64}  = 2*pi*rand(mesh.nInsideNodes)
    phi::Vector{Float64}    = pi*rand(mesh.nInsideNodes)
    if random
        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]

            # Initialize a random magnetization field
                m[:,nd] = [ sin(theta[i])*cos(phi[i]),
                            sin(theta[i])*sin(phi[i]),
                            cos(theta[i])]
        end
    else # Initialize magnetization with direction
        # normalize direction
        direction = direction./norm(direction)
        m[:,mesh.InsideNodes] .= direction
    end

    return m
end # Initialize magnetization field

# Find new magnetization direction iteratively
function iterate(M::Vector{Float64},H::Vector{Float64},Hold::Vector{Float64},Heff::Vector{Float64},damp::Float64,giro::Float64,dt::Float64,precession::Bool)
    #=
        Finds the new magnetization direction given:
         .the current magnetization direction (M)
         .the magnetic field H tilde (H) for current and previous iteration
         .the effective field (Heff) for the current iteration

        H tilde = Heff + damp*cross(m,Heff)
    =#
    Mnew::Vector{Float64} = zeros(length(M))

    d = giro*dt/2
    
    # 1. Calculate H n + 1/2 from the previous two magnetic fields (H n and H n-1)
    h12 = 3/2 .*H - 0.5 .*Hold

    aux = M
    err::Float64 = 1.0
    att::Int32 = 0
    while err > 1e-6
        att += 1

        # 2. M (n+1) from M and H(n+1/2)
        mat::Matrix{Float64} = [1 d*h12[3] -d*h12[2];
                                -d*h12[3] 1 d*h12[1];
                                d*h12[2] -d*h12[1] 1];

        Mnew = mat\(M-d.*cross(M,h12))

        # 3. M (n + 1/2)
        m12 = 0.5*(M+Mnew)

        # 4. Calculate H (n + 1/2) from the M (n + 1/2)
        h12 = precession*Heff + damp.*cross(m12,Heff)

        # Difference between the new guess and the previous guess
        # err = norm(Mnew-aux)
        err = maximum(abs.(Mnew-aux))
        
        # println(err)

        # Store new value of M (n+1)
        aux = Mnew

        if att > 50
            println("Magnetization iteration stuck")
            break
        end

    end # End of loop to find new magnetization direction
    
    return Mnew
end # Iteration function to find new magnetization direction

function LandauLifshitz(mesh, m::Matrix{Float64}, Heff::Matrix{Float64}, H::Matrix{Float64}, Hext::Matrix{Float64},
                        Ms::Float64, Aexc::Float64, Aan::Float64, uan::Vector{Float64},
                        dt::Float64, totalTime::Float64, Vn::Vector{Float64}, nodeVolume::Vector{Float64}, scl::Float64, fixed::Vector{Int32}, free::Vector{Int32}, AD, A,
                        damp::Float64, giro::Float64, precession::Bool=true, maxAtt::Int32=10_000,maxTorque::Float64=1e-5)
    #=
        mesh
        
        Heff        Matrix 3 by mesh.nv, total magnetic field on each node
        H           Matrix 3 by mesh.nv, Cross product of m and Heff
        Hext        Matrix, 3 by mesh.nv, external field on each mesh node
        Ms          Magnetic saturation
        Aexc        Exchange parameter
        Aan         Anisotropy constant
        uan         Anisotropy direction

        dt          Time step
        totalTime   Total time for spin dynamics
        
        Vn          vector
        nodeVolume  vector
        fixed       nodes for the boundary condition magnetic scalar potential = 0
        free        nodes without imposed conditions
        AD          stiffness matrix of the demag field
        
        A           Stiffness matrix of the exchange field
        
        damp        damping parameter
        giro        giromagnetic ratio
        precession  Consider or not spin precession
        maxAtt      Maximum number of attempts
    =#
    
    mu0::Float64 = pi*4e-7  # vacuum magnetic permeability

    # Store initial conditions
    Hold::Matrix{Float64} = H

    M_avg::Matrix{Float64} = zeros(3,maxAtt)   
    E_time::Vector{Float64} = zeros(maxAtt)
    torque_time::Vector{Float64} = zeros(maxAtt)

    t::Float64 = 0 # Initial time
    it::Int32 = 0
    for att in 1:maxAtt
        it = att # Store the number of time iterations 
        t += dt  # Update the time

        println(1e9*t," ns")

        # Rotate magnetization
        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            m[:,nd] = iterate(m[:,nd],H[:,i],Hold[:,i],Heff[:,i],damp,giro,dt,precession)
        end

        # New magnetic field
        Hold = H

        # Applied field (T)
        # Hext::Matrix{Float64} = zeros(3,mesh.nInsideNodes) .+ ~.*Hap

        # Demagnetizing field (T)
        global Hd = demagField(mesh,fixed,free,AD,m)
        
        # Exchange field (T)
        global Hexc = -2*Aexc.* (A*m[:,mesh.InsideNodes]')'

        # Anisotropy field (T)
        global Han = zeros(3,mesh.nInsideNodes)
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
        Heff = Hext + Hd + Hexc + Han
        
        # Torque term
        H = zeros(3,mesh.nInsideNodes) 
        for i in 1:mesh.nInsideNodes
            H[:,i] = Heff[:,i] + damp.*cross(m[:,i],Heff[:,i])
        end

        # Energy density
        global Eext = 0
        global Ed   = 0
        global Eexc = 0
        global Ean  = 0

        for i in 1:mesh.nInsideNodes
            Eext -= dot(m[:,i],Hext[:,i])
            Ed   -= 0.5*dot(m[:,i],Hd[:,i])
            Eexc -= 0.5*dot(m[:,i],Hexc[:,i])
            Ean  -= 0.5*dot(m[:,i],Han[:,i])
        end
        global E = mu0*Ms*(Eext + Ed + Eexc + Ean)
        E_time[it] = E # Store the energy for each time step

        # Average torque
        dtau::Float64 = 0
        for i = 1:mesh.nInsideNodes
            dtau += norm(cross(m[:,mesh.InsideNodes[i]],Heff[:,i]))
        end
        torque_time[it] = dtau/mesh.nInsideNodes
    
        # Average magnetization
        M_avg[:,it] = mean(m[:,mesh.InsideNodes],2)

        # Stop at totalTime ns
        if totalTime > 0 && 1e9 * t > totalTime
            break
        
        # Stop if torque term is small
        elseif dtau/mesh.nInsideNodes < maxTorque 
            break
        end

    end # End of time iteration

    # Remove excess zeros
    M_avg       = M_avg[:,1:it]
    E_time      = E_time[1:it]
    torque_time = torque_time[1:it]

    time::Vector{Float64} = 1e9 .*(1:it)
    return m, time, E_time, torque_time, M_avg, Hd, Hexc, Han, Heff, E, mu0*Ms*Eext, mu0*Ms*Ed, mu0*Ms*Eexc, mu0*Ms*Ean
end


# Time dependent micromagnetic simulation
function runTimeDependent(meshSize=0,localSize=0,showGmsh=false,saveMesh=false)

    #=
        Micromagnetic simulation of a plate with dimensions L, replicating the results seen in Fig. 2
        of this article - https://doi.org/10.1109/TMAG.2008.2001666
    =#

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

    # Comment this if you want to check the mesh size before running
    # return 

    # View the mesh using Julia instead of Gmsh
    # viewMesh(mesh)

    # ------------- Micromagnetic simulation --------------
    
    # Initial magnetization value | true -> random field
    m::Matrix{Float64} = initialMagnetization(mesh,false,[1,0,0])

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

    # ---- Initial magnetic field ----
    
    # Applied field (T)
    Hext::Matrix{Float64} = zeros(3,mesh.nInsideNodes) .+ mu0.*Hap

    # Demagnetizing field (T)
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
    
    # plotHField(mesh,Hd)
    # plotHField(mesh,Hexc)
    # plotHField(mesh,Han)

    # Effective field
    Heff::Matrix{Float64} = Hext + Hd + Hexc + Han
    
    # Torque term
    H::Matrix{Float64} = zeros(3,mesh.nInsideNodes) 
    for i in 1:mesh.nInsideNodes
        H[:,i] = Heff[:,i] + damp.*cross(m[:,i],Heff[:,i])
    end

    # Energy density
    Eext::Float64  = 0
    Ed::Float64    = 0
    Eexc::Float64  = 0
    Ean::Float64   = 0

    for i in 1:mesh.nInsideNodes
        Eext -= dot(m[:,i],Hext[:,i])
        Ed   -= 0.5*dot(m[:,i],Hd[:,i])
        Eexc -= 0.5*dot(m[:,i],Hexc[:,i])
        Ean  -= 0.5*dot(m[:,i],Han[:,i])
    end
    E::Float64 = mu0*Ms*(Eext + Ed + Eexc + Ean)

    # -------- time iteration ---------
    m, time::Vector{Float64}, 
    E_time::Vector{Float64}, torque_time::Vector{Float64}, M_avg::Matrix{Float64}, 
    Hd, Hexc, Han, Heff,
    E, Eext, Ed, Eexc, Ean = LandauLifshitz(mesh, m, Heff, H, Hext,
                                            Ms, Aexc, Aan, uan, dt, totalTime, 
                                            Vn, nodeVolume, scl, fixed, free, AD, A, 
                                            damp, giro, precession, maxAtt)

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
    save("M_time.png",fig)
end # end of time dependent micromagnetic simulation

# Steepest descent, micromagnetic simulation
function minimalEnergy(meshSize=0,localSize=0,showGmsh=false,saveMesh=false)
    #=
        Micromagnetic simulation without time dynamics, only energy minimum.
        The energy minimization follows the steepest descent algorithm from
        https://doi.org/10.1063/1.4862839
                    and
        https://doi.org/10.1063/1.4896360
    =#

    # Constants
    mu0::Float64 = pi*4e-7          # vacuum magnetic permeability
    giro::Float64 = 2.210173e5 /mu0 # Gyromagnetic ratio (rad T-1 s-1)
    dt::Float64 = 0.028/giro        # Time step in reduced units (seconds per gyro)
    totalTime::Float64 = 0.0        # Total time of spin dynamics simulation (ns)
    damp::Float64 = 1               # Damping parameter (dimensionless [0,1])
    precession::Bool = false        # Include precession or not

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

    # Comment this if you want to check the mesh size before running
    # return 

    # View the mesh using Julia instead of Gmsh
    # viewMesh(mesh)

    # ------------- Micromagnetic simulation --------------
    
    # Initial magnetization value | true -> random field
    m::Matrix{Float64} = initialMagnetization(mesh,false,[1,0,0])

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

    # ---- Initial magnetic field ----
    
    # Applied field (T)
    Hext::Matrix{Float64} = zeros(3,mesh.nInsideNodes) .+ mu0.*Hap

    # Demagnetizing field (T)
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
    
    # plotHField(mesh,Hd)
    # plotHField(mesh,Hexc)
    # plotHField(mesh,Han)

    # Effective field
    Heff::Matrix{Float64} = Hext + Hd + Hexc + Han
    
    # Torque term
    H::Matrix{Float64} = zeros(3,mesh.nInsideNodes) 
    for i in 1:mesh.nInsideNodes
        H[:,i] = Heff[:,i] + damp.*cross(m[:,i],Heff[:,i])
    end

    # Energy density
    Eext::Float64  = 0
    Ed::Float64    = 0
    Eexc::Float64  = 0
    Ean::Float64   = 0

    for i in 1:mesh.nInsideNodes
        Eext -= dot(m[:,i],Hext[:,i])
        Ed   -= 0.5*dot(m[:,i],Hd[:,i])
        Eexc -= 0.5*dot(m[:,i],Hexc[:,i])
        Ean  -= 0.5*dot(m[:,i],Han[:,i])
    end
    E::Float64 = mu0*Ms*(Eext + Ed + Eexc + Ean)

    # Initial relaxation
    # Use the landau lifshitz equation to relax the spin system until the torque term is small
    maxTorque::Float64 = 1e-5

    m, _, 
    _, _, _, 
    _, _, _, Heff,
    E, _, _, _, _ = LandauLifshitz(mesh, m, Heff, H, Hext,
                                            Ms, Aexc, Aan, uan, dt, totalTime, 
                                            Vn, nodeVolume, scl, fixed, free, AD, A, 
                                            damp, giro, precession, maxAtt,1e-5)


    # ============ Steepest descent =============

    # Torque term
    H = zeros(3,mesh.nInsideNodes) 
    for i in 1:mesh.nInsideNodes
        H[:,i] = cross(m[:,i],Heff[:,i])
    end

    # Rotate magnetization
    for i in 1:mesh.nInsideNodes
        nd = mesh.InsideNodes[i]
        m[:,nd] = iterate(m[:,nd],H[:,i],H[:,i],Heff[:,i],damp,giro,dt,precession)
    end

    # M_avg::Matrix{Float64} = zeros(3,maxAtt)   
    # E_time::Vector{Float64} = zeros(maxAtt)
    # # torque_time::Vector{Float64} = zeros(maxAtt)
    # for att in 1:maxAtt

    #     # Save previous iteration
    #     Heff_old::Matrix{Float64} = Heff;
    #     Hold::Matrix{Float64} = H;

    #     # New magnetic field
        
    #     # Demagnetizing field (T)
    #     Hd = demagField(mesh,fixed,free,AD,m)
        
    #     # Exchange field (T)
    #     Hexc = -2*Aexc.* (A*m[:,mesh.InsideNodes]')'

    #     # Anisotropy field (T)
    #     Han = zeros(3,mesh.nInsideNodes)
    #     for i in 1:mesh.nInsideNodes
    #         nd = mesh.InsideNodes[i]
    #         Han[:,i] = 2*Aan/Ms*dot(m[:,nd],uan).*uan
    #     end
        
    #     # Convert to proper unis
    #     @simd for i in 1:3
    #         Hd[i,:]     .*= mu0*Ms./Vn
    #         Hexc[i,:]   ./= Ms*scl^2 .*nodeVolume
    #     end
        
    #     # Effective field
    #     Heff::Matrix{Float64} = Hext + Hd + Hexc + Han

    #     # Torque term
    #     H = zeros(3,mesh.nInsideNodes) 
    #     for i in 1:mesh.nInsideNodes
    #         H[:,i] = cross(m[:,i],Heff[:,i])
    #     end

    #     Eext = Ed = Eexc = Ean = 0
    #     for i in 1:mesh.nInsideNodes
    #         Eext -= dot(m[:,i],Hext[:,i])
    #         Ed   -= 0.5*dot(m[:,i],Hd[:,i])
    #         Eexc -= 0.5*dot(m[:,i],Hexc[:,i])
    #         Ean  -= 0.5*dot(m[:,i],Han[:,i])
    #     end
    #     E = mu0*Ms*(Eext + Ed + Eexc + Ean)

    #     # Get a new time step
    #     snN::Float64  = 0
    #     snD::Float64  = 0;
    #     snD2 = 0;
    #     for i = 1:mesh.nInsideNodes
    #         nd = mesh.InsideNodes(i);
            
    #         sn = m(:,nd)-mOld(:,nd);
            
    #         gn2 = cross(m(:,nd),cross(m(:,nd),Heff(:,i)));
    #         gn1 = cross(mOld(:,nd),cross(mOld(:,nd),Heff_old(:,i)));

    #         snN  = snN + dot(sn,sn);
    #         snD  = snD + dot(sn,gn2-gn1);
    #         snD2 = snD2 + dot(gn2-gn1,gn2-gn1);
    #     end

    #     tau1 = snN/snD;
    #     tau2 = snD/snD2;

    #     if mod(att,2) > 0
    #         dt = tau1;
    #     else
    #         dt = tau2;
    #     end

    #     if isnan(dt)
    #         disp("tau is nan")
    #         return
    #     end

    #     % New magnetization and check deviation
    #     dm = 0;
    #     for i = 1:mesh.nInsideNodes
    #         nd = mesh.InsideNodes(i);

    #         [m(:,nd),mOld(:,nd)] = nextM(m(:,nd),Heff(:,i),dt);
    #         dm = max(dm,norm(m(:,nd)-mOld(:,nd)));
    #     end

    #     M_avg(:,att) = mean(m(:,mesh.InsideNodes),2);

    #     if dm < maxDeviation
    #         disp("M is stable")
    #         break
    #     end
    # end % End of iterative steepest descent 

    # if att == maxAtt
    #     disp("Reached maximum attempts")
    # end

    # M_avg = M_avg(:,1:att);
    # E = E(1:att);

    # fig = Figure()
    # ax = Axis(  fig[1,1], 
    #             xlabel = "Time (ns)", 
    #             ylabel = "<M> (kA/m)",
    #             title = "Micromagnetic simulation")

    # scatter!(ax,time,E_time, label = "Energy")
    # scatter!(ax,time,torque_time, label = "torque")
    # axislegend() # position = :rt

    # wait(display(fig))
    # save("E_time.png",fig)
end

# Run micromagnetic simulation with energy minimization by steepest descent
minimalEnergy(200,5,false,false)

# Run time dependent micromagnetic simulation
# runTimeDependent(200,5,false,false)

