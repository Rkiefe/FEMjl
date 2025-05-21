#=
    This file holds the Finite-Element related logic

    abcd                 -> gets the linear basis function, f = a + bx + cy + dz
    stiffnessMatrix      -> the global stiffness matrix (sparse)
    lagrange             -> Calculates the volume integral of the basis function,
                            used for the Lagrange multiplier technique
    BoundaryIntegral     -> The surface integral of a vector field dot surface normal
    demagField           -> Calculates the demagnetizing field of a magnetization field
    mean                 -> Replicates the matlab 'mean' function
    CstiffnessMatrix     -> Calculates the local stiffness matrix in C++
    localstiffnessmatrix -> Dense, local stiffness matrix in Julia
=#

using LinearAlgebra, SparseArrays

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

# Dense, local stiffness matrix
function localStiffnessMatrix(mesh::MESH,f::Vector{Float64})
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


# Sparse, global stiffness matrix
function stiffnessMatrix(mesh::MESH,f::Vector{Float64}=ones(mesh.nt))
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
function lagrange(mesh::MESH)
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
function BoundaryIntegral(mesh::MESH,F,shell_id)
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

# Demagnetizing field
function demagField(mesh::MESH,fixed::Vector{Int32},free::Vector{Int32},A,m::Matrix{Float64})
    #= 
        Calculates the demagnetizing field attributed to a magnetization field
        using FEM and a bounding shell

        Inputs:
            mesh    (mesh data)
            fixed   (nodes for the boundary condition magnetic scalar potential = 0)
            free    (nodes without imposed conditions)
            A       (Stiffness matrix)
            m       (Magnetization vector field, 3 by mesh.nv)
    =#
    
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

# ============= All-purpose functions ==================
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


# ============= C/C++ interop ==============
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
