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

# Mean function
function mean(arr::Vector,dimension=1)
    m::Real = 0
    for x in arr
        m += x
    end

    return m/length(arr)
end # End of mean for vectors

function mean(arr::Matrix,dimension=1)
    if dimension == 1 
        d = size(arr,1)
        m = 0 .*arr[1,:]
        for i in 1:d
            m .+= arr[i,:]
        end

    else
        d = size(arr,2)
        m = 0 .*arr[:,1]
        for i in 1:d
            m .+= arr[:,i]
        end
    end

    return m./d
end # End of mean for 2D matrices


# Demagnetizing field
function demagField(mesh,fixed::Vector{Int32},free::Vector{Int32},A,m::Matrix{Float64})
    # Load vector
    RHS::Vector{Float64} = zeros(mesh.nv)
    for ik in 1:mesh.nInside
        k = mesh.InsideElements[ik]
        nds = mesh.t[1:4,k]

        # Average magnetization in the element
        aux = mean(m[:,nds],2)
        for i in 1:4
            _,b,c,d = abcd(mesh.p,nds,nds[i])
            RHS[nds[i]] += mesh.VE[k]*dot([b,c,d],aux)
        end
    end

    u::Vector{Float64} = zeros(mesh.nv)
    u[free] = A[free,free]\RHS[free]

    # Demagnetizing field | Elements
    Hde::Matrix{Float64} = zeros(3,mesh.nInside)
    for ik in 1:mesh.nInside
        k = mesh.InsideElements[ik]
        nds = mesh.t[:,k] # all nodes of that element

        # Sum the contributions
        for ind in 1:length(nds)
            nd = nds[ind]

            # obtain the element parameters
            _,bi,ci,di = abcd(mesh.p,nds,nd)

            Hde[1,ik] -= u[nd]*bi
            Hde[2,ik] -= u[nd]*ci
            Hde[3,ik] -= u[nd]*di
        end
    end

    # Demagnetizing field | Nodes
    Hd::Matrix{Float64} = zeros(3,mesh.nv)
    for ik in 1:mesh.nInside
        k = mesh.InsideElements[ik]
        nds = mesh.t[:,k]
        Hd[:,nds] .+= mesh.VE[k]*Hde[:,ik]
    end
    Hd = Hd[:,mesh.InsideNodes]

    return Hd
end

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

function main(meshSize=0,localSize=0,showGmsh=true,saveMesh=false)
    
    # Constants
    mu0 = pi*4e-7                   # vacuum magnetic permeability
    giro = 2.210173e5 /mu0          # Gyromagnetic ratio (rad T-1 s-1)
    dt::Float64 = 0.028/giro        # Time step in reduced units (seconds per gyro)
    damp::Float64 = 0.1             # Damping parameter (dimensionless [0,1])
    precession::Bool = true         # Include precession or not

    # Dimension of the magnetic material (rectangle)
    L::Vector{Float64} = [100,100,5]
    scl::Float64 = 1e-9                 # scale of the geometry | (m -> nm)
    
    # Conditions
    Ms   = 860e3                       # Magnetic saturation (A/m)
    Aexc = 13e-12                      # Exchange   (J/m)
    Aan  = 0                           # Anisotropy (J/m3)
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

    # Store initial conditions
    Hold::Matrix{Float64} = H
    Eold::Float64 = E

    t::Float64 = 0 # Initial time

    M_avg::Matrix{Float64} = zeros(3,maxAtt)   
    E_time::Vector{Float64} = zeros(maxAtt)
    torque_time::Vector{Float64} = zeros(maxAtt)


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
        Eold = E

        # Applied field (T)
        # Hext::Matrix{Float64} = zeros(3,mesh.nInsideNodes) .+ mu0.*Hap

        # Demagnetizing field (T)
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
        Heff = Hext + Hd + Hexc + Han
        
        # Torque term
        H = zeros(3,mesh.nInsideNodes) 
        for i in 1:mesh.nInsideNodes
            H[:,i] = Heff[:,i] + damp.*cross(m[:,i],Heff[:,i])
        end

        # Energy density
        Eext = 0
        Ed   = 0
        Eexc = 0
        Ean  = 0

        for i in 1:mesh.nInsideNodes
            Eext -= dot(m[:,i],Hext[:,i])
            Ed   -= 0.5*dot(m[:,i],Hd[:,i])
            Eexc -= 0.5*dot(m[:,i],Hexc[:,i])
            Ean  -= 0.5*dot(m[:,i],Han[:,i])
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

        # Stop at 0.4 ns
        if 1e9 * t > 0.4
            break
        end

        # Check if average magnetization is stable
        # if it > 20
        #     div = norm(M_avg[:,it]-M_avg(:,it-1))
        #     println("t (ns): ",t*1e9," div(x100): ",100*div)
        #     if div < maxTau
        #         println("Average magnetization is stable")
        #         break
        #     end
        # end

    end # End of time iteration

    # Remove excess zeros
    M_avg       = M_avg[:,1:it]
    E_time      = E_time[1:it]
    torque_time = torque_time[1:it]

    time::Vector{Float64} = 1e9*dt .* (1:it)

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
end # end of main

meshSize = 200
localSize = 5
showGmsh = false
saveMesh = false

main(meshSize,localSize,showGmsh,saveMesh)

