#=
    3D Finite element implementation of the Landau-Lifshitz equation with a semi-implicit
    time step, based on https://doi.org/10.1109/TMAG.2008.2001666 (Oriano 2008)

    The energy is minimized by a steepest descent algorithm from 

    https://doi.org/10.1063/1.4862839
                and
    https://doi.org/10.1063/1.4896360

=#

include("LandauLifshitz.jl")  # For the LL solver

# Next step in magnetization by steepest descent
function nextM(M::Vector{Float64},Heff::Vector{Float64},dt::Float64)
    # Semi-implicit time step
    d::Float64 = dt/2;
    h12 = cross(M,Heff)

    mat = [1 d*h12[3] -d*h12[2];
           -d*h12[3] 1 d*h12[1];
           d*h12[2] -d*h12[1] 1]

    Mnew = mat\(M-d*cross(M,h12))

    return Mnew, M
end # New magnetization with steepest descent

# Energy minimization by steepest descent
function steepestDescent(mesh, scl::Float64, m::Matrix{Float64}, Ms::Float64, Aexc::Float64, Aan::Float64, uan::Vector{Float64}, Hap::Vector{Float64}, giro::Float64, maxTorque::Float64, maxAtt::Int32=10_000)
    #=
        Minimizes the energy of the system by a steepest descent approach
    =#

    mu0::Float64 = pi*4e-7          # vacuum magnetic permeability

    # -- Finite element preliminaries --
            
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
        A = spzeros(mesh.nv,mesh.nv)
        begin # Make a local scope to keep the workspace clean
            
            Ak::Matrix{Float64} = zeros(4*4,mesh.nt)
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

    # -- Magnetic fields --
        
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
            H[:,i] = cross(m[:,nd],Heff[:,i])
        end

    # -- Time step --

    # Make one iteration using the same approach as with LandauLifshitz()
    mOld::Matrix{Float64} = deepcopy(m)
    for i in 1:mesh.nInsideNodes
        nd = mesh.InsideNodes[i]
        m[:,nd] = timeStep(mOld[:,nd],H[:,i],H[:,i],Heff[:,i],0.03,1.0,1.0,false)
    end

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

    div::Float64 = maxTorque + 1
    it::Int32 = 0       # Iteration step
    while div > maxTorque && it < maxAtt
        it += 1

        if mod(it,50) < 1
            println("SD iteration: ",it,"; Average torque: ",div)
        end

        # Store previous magnetic field
        HeffOld::Matrix{Float64} = deepcopy(Heff)
        Hold::Matrix{Float64} = deepcopy(H)

        # -- New magnetic field --
            
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
                H[:,i] = cross(m[:,nd],Heff[:,i])
            end

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
        dtau = 0
        for i = 1:mesh.nInsideNodes
            dtau += norm(cross(m[:,mesh.InsideNodes[i]],Heff[:,i]))
        end
        torque_time[it] = dtau/mesh.nInsideNodes

        # Calculate the time step
        snN::Float64  = 0
        snD::Float64  = 0
        snD2::Float64 = 0
        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            
            sn::Vector{Float64} = m[:,nd] - mOld[:,nd]
            
            gn2::Vector{Float64} = cross(m[:,nd], cross(m[:,nd], Heff[:,i]))
            gn1::Vector{Float64} = cross(mOld[:,nd], cross(mOld[:,nd],HeffOld[:,i]))

            snN  += dot(sn,sn)
            snD  += dot(sn,gn2-gn1)
            snD2 += dot(gn2-gn1,gn2-gn1)
        end

        tau1 = snN/snD;
        tau2 = snD/snD2;

        # Alternate between time steps
        dt::Float64 = 0
        if mod(it,2) > 0
            dt = tau1
        else
            dt = tau2
        end

        # New magnetization direction
        for i in 1:mesh.nInsideNodes
            nd = mesh.InsideNodes[i]
            m[:,nd], mOld[:,nd] = nextM(m[:,nd],Heff[:,i],dt)
        end

        # Average magnetization
        M_avg[:,it] = mean(m[:,mesh.InsideNodes],2)

        # Check stability by evaluating the torque term
        div = torque_time[it]
        if torque_time[it] < maxTorque
            println("Torque is small, exiting relax()")
        end

    end # End of energy minimization by steepest descent
    
    # Remove excess zeros
    M_avg       = M_avg[:,1:it]
    E_time      = E_time[1:it]
    torque_time = torque_time[1:it]

    return m, Heff, M_avg, E_time, torque_time, Hd, Hexc, Han, E, Ed, Eexc, Ean

end # End of steepestDescent
