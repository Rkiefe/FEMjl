/*
	Calculates the dense, local stiffness matrix
	Also contains a wrapper to be called by Julia
*/

#include <iostream>	
#include <vector>
#include <omp.h>

// Eigen for linear algebra
#include <C:\Users\rkiefe\Documents\eigen\Eigen\Dense>

// FEM basis function
Eigen::Vector4d abcd(Eigen::MatrixXd p,Eigen::Vector4i nodes,int nd){
	
	int nds[3] = {0,0,0};		// All other nodes of the element
	{ // Get nodes different than nd
	int n = 0;
	for(int i = 0; i<4; i++){
		if(nodes(i)!=nd){
			nds[n] = nodes(i);
			n++;
		}
	}
	} // Get nodes different than nd

	// Build the matrix equation for the a b c d
	Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
	M.row(0) << 1,p(0,nd),p(1,nd),p(2,nd);
	M.row(1) << 1,p(0,nds[0]),p(1,nds[0]),p(2,nds[0]);
	M.row(2) << 1,p(0,nds[1]),p(1,nds[1]),p(2,nds[1]);
	M.row(3) << 1,p(0,nds[2]),p(1,nds[2]),p(2,nds[2]);

	Eigen::Vector4d b(1.0,0,0,0);
	Eigen::Vector4d r = M.colPivHouseholderQr().solve(b);

	return r;
} // FEM basis function

void localStiffnessMatrix(Eigen::Ref<Eigen::MatrixXd> Ak, Eigen::Ref<Eigen::MatrixXd> p, Eigen::Ref<Eigen::MatrixXi> t, double* VE, double* mu){
	// #pragma omp parallel for
	for(int k = 0; k<t.cols(); k++){
		Eigen::Vector4d b,c,d;
		for(int i = 0; i<4; i++){
			Eigen::Vector4d r = abcd(p,t.col(k),t(i,k));
			b(i) = r[1];
			c(i) = r[2];
			d(i) = r[3]; 
		}
		Eigen::Matrix4d aux = VE[k]*mu[k]*(b*b.transpose() + c*c.transpose() + d*d.transpose());
		Ak.col(k) = Eigen::Map<Eigen::VectorXd>(aux.data(), 16);
	}
} // Local stiffness matrix


extern "C"{
	// Function called from Julia
	void abcd_julia(double* pJL, int* nodes, int nd, int nv, double* result){
		// Convert the pointer array to Eigen matrix
		Eigen::Map<Eigen::MatrixXd> p(pJL,3,nv);
		Eigen::Map<Eigen::Vector4i> nds(nodes);
		
		// Run C++ abcd
		Eigen::Vector4d r = abcd(p,nds,nd);

		// Update the output
		result[0] = r[0];
		result[1] = r[1];
		result[2] = r[2];
		result[3] = r[3];
	}

	void stiffnessMatrix(double* AkJL, double* pJL, int* tJL, int nv, int nt, double* VE, double* mu){
		// Convert the pointer array to Eigen matrix
		Eigen::Map<Eigen::MatrixXd> Ak(AkJL,16,nt);
		Eigen::Map<Eigen::MatrixXd> p(pJL,3,nv);
		Eigen::Map<Eigen::MatrixXi> t(tJL,4,nt);

		// Run C++ local stiffness matrix
		localStiffnessMatrix(Ak,p,t,VE,mu);
	}

} // extern C

