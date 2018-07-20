// Author: Akash Patel (apatel435@gatech.edu), Areeb Mehmood (amehmood7@gatech.edu)

// genPhiMatrixAsFile
// Purpose: Determine phi vectors for each input pose
//   This phi will be used for finding beta (weights of actual robot) via gradient descent or some other method
//
// Input: Ideal beta={mi, MXi, MYi, ...}, krang urdf model, perturbation value,
//   potentially unbalanced data points (q/poses) as a file,
// Output: Phi matrix as a file

// Overall Input: Poses in {heading, qBase, etc.} format
// Intermediary Input/Output Flow:
// Input Pose File -> Dart Poses -> Opt Dart Poses -> Phi Matrix
// Phi Matrix -> Converged Beta
//

// TODO: Perform C++ warning checks
// TODO: Check for memory leaks (valgrind)

#include <dart/dart.hpp>
#include <dart/utils/urdf/urdf.hpp>
#include <iostream>
#include <fstream>
#include <nlopt.hpp>
#include <cmath>

using namespace std;
using namespace dart::common;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::math;

#define MAXBUFSIZE ((int) 1e6)

Eigen::MatrixXd readInputFileAsMatrix(string inputFilename);
Eigen::MatrixXd allInitq;
Eigen::MatrixXd allInitqdot;
Eigen::MatrixXd allInitqdotdot;
Eigen::MatrixXd allInitTorque;
Eigen::MatrixXd allInitM;
Eigen::MatrixXd allInitCg;

int numDataPts;

int genPhiMatrixAsFile() {

/*============================================================================================*/
/*====================================Read in text files======================================*/
/*============================================================================================*/
	string inputQFilename = "../../data/qWaistData.txt";
    string inputQdotFilename = "../../data/dqWaistData.txt";
    string inputQdotdotFilename = "../../data/ddqWaistData.txt";
    string inputTorqueFilename = "../../data/torqueWaistData.txt";
    string inputMassMatrixFilename = "../../data/mWaistData.txt";
    string inputCgFilename = "../../data/cgWaistData.txt";

    try{
        cout << "Reading input q ...\n";
        allInitq = readInputFileAsMatrix(inputQFilename);
        cout << "|-> Done\n";

        cout << "Reading input qdot ...\n";
        allInitqdot = readInputFileAsMatrix(inputQdotFilename);
        cout << "|-> Done\n";

        cout << "Reading input qdotdot ...\n";
        allInitqdotdot = readInputFileAsMatrix(inputQdotdotFilename);
        cout << "|-> Done\n";

        cout << "Reading input torque ...\n";
        allInitTorque = readInputFileAsMatrix(inputTorqueFilename);
        cout << "|-> Done\n";

        cout << "Reading input M ...\n";
        allInitM = readInputFileAsMatrix(inputMassMatrixFilename);
        cout << "|-> Done\n";

        cout << "Reading input Cg ...\n";
        allInitCg = readInputFileAsMatrix(inputCgFilename);
        cout << "|-> Done\n";

    } catch (exception& e) {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }

/*============================================================================================*/
/*====================================Instantiate Variables===================================*/
/*============================================================================================*/

    double perturbedValue = 1e-8;
    // Instantiate "ideal" robot
    cout << "Creating ideal beta vector ...\n";
    dart::utils::DartLoader loader;
    dart::dynamics::SkeletonPtr idealRobot = loader.parseSkeleton("/home/munzir/dart_test/09-URDF/KrangWaist/krang_fixed_base.urdf");
    idealRobot->setGravity(Eigen::Vector3d (0.0, -9.81, 0.0));
    
    // Get ideal beta
    // Beta Definition/Format
    // mi, mxi, myi, mzi for each body

    int bodyParams = 10;
    int numBodies = idealRobot->getNumBodyNodes(); //returns 18
    dart::dynamics::BodyNodePtr bodyi;
    string namei;
    //These are our 10 parameters, + 3 variables xi, yi, zi
	double mi;
    double xi, xMi;
    double yi, yMi;
    double zi, zMi;
    double ixx, ixy, ixz, iyy, iyz, izz;

    int numBetaVals = (numBodies-1)*bodyParams;
    Eigen::MatrixXd betaParams(1, numBetaVals);

	//Fill betaParams array with values from URDF (masses, inertias) 
    for (int i = 1; i < numBodies; i++) {
        bodyi = idealRobot->getBodyNode(i);

        namei = bodyi->getName();
        mi = bodyi->getMass();
        xMi = mi * bodyi->getLocalCOM()(0);
        yMi = mi * bodyi->getLocalCOM()(1);
        zMi = mi * bodyi->getLocalCOM()(2);
        bodyi->getMomentOfInertia (ixx,iyy,izz,ixy,ixz,iyz);

        betaParams(0, (i-1) * bodyParams + 0) = mi;
        betaParams(0, (i-1) * bodyParams + 1) = xMi;
        betaParams(0, (i-1) * bodyParams + 2) = yMi;
        betaParams(0, (i-1) * bodyParams + 3) = zMi;
        betaParams(0, (i-1) * bodyParams + 4) = ixx;
        betaParams(0, (i-1) * bodyParams + 5) = iyy;
        betaParams(0, (i-1) * bodyParams + 6) = izz;
        betaParams(0, (i-1) * bodyParams + 7) = ixy;
        betaParams(0, (i-1) * bodyParams + 8) = ixz;
        betaParams(0, (i-1) * bodyParams + 9) = iyz; 
    }
    // cout << betaParams << endl;

    cout << "|-> Done\n";

    //Save beta parameters
    ofstream betafile;
    betafile.open ("betaparameters.txt");
    betafile<< betaParams.transpose()<<endl;
    betafile.close();

/*============================================================================================*/
/*====================================Load array of Robot=====================================*/
/*============================================================================================*/

    cout << "Creating robot array ...\n";
	//load robots into fwdPertRobotArray and revPertRobotArray
    dart::dynamics::SkeletonPtr fwdPertRobotArray[numBetaVals];
    dart::dynamics::SkeletonPtr revPertRobotArray[numBetaVals];
    for(int i=0; i<numBetaVals; i++) {
        fwdPertRobotArray[i] = idealRobot->clone();
        revPertRobotArray[i] = idealRobot->clone();
    }

    //Perturb all Beta values in the forward direction
    for(int i=1; i<numBodies; i++) { //for 17 loops
	    fwdPertRobotArray[(i-1)*bodyParams + 0]->getBodyNode(i)->setMass(mi + perturbedValue);
	    fwdPertRobotArray[(i-1)*bodyParams + 1]->getBodyNode(i)->setLocalCOM(Eigen::Vector3d(xi + perturbedValue, yi, zi));
	    fwdPertRobotArray[(i-1)*bodyParams + 2]->getBodyNode(i)->setLocalCOM(Eigen::Vector3d(xi, yi + perturbedValue, zi));
	    fwdPertRobotArray[(i-1)*bodyParams + 3]->getBodyNode(i)->setLocalCOM(Eigen::Vector3d(xi, yi, zi + perturbedValue));
	    fwdPertRobotArray[(i-1)*bodyParams + 4]->getBodyNode(i)->setMomentOfInertia(ixx + perturbedValue, iyy, izz, ixy, ixz, iyz);
	    fwdPertRobotArray[(i-1)*bodyParams + 5]->getBodyNode(i)->setMomentOfInertia(ixx, iyy + perturbedValue, izz, ixy, ixz, iyz);
	    fwdPertRobotArray[(i-1)*bodyParams + 6]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz + perturbedValue, ixy, ixz, iyz);
	    fwdPertRobotArray[(i-1)*bodyParams + 7]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz, ixy + perturbedValue, ixz, iyz);
	    fwdPertRobotArray[(i-1)*bodyParams + 8]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz, ixy, ixz + perturbedValue, iyz);
	    fwdPertRobotArray[(i-1)*bodyParams + 9]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz, ixy, ixz, iyz + perturbedValue);  
	}

	//Perturb all Beta values in the reverse direction
    for(int i=1; i<numBodies; i++) {
        revPertRobotArray[(i-1)*bodyParams + 0]->getBodyNode(i)->setMass(mi - perturbedValue);
        revPertRobotArray[(i-1)*bodyParams + 1]->getBodyNode(i)->setLocalCOM(Eigen::Vector3d(xi -  perturbedValue, yi, zi));
        revPertRobotArray[(i-1)*bodyParams + 2]->getBodyNode(i)->setLocalCOM(Eigen::Vector3d(xi, yi - perturbedValue, zi));
        revPertRobotArray[(i-1)*bodyParams + 3]->getBodyNode(i)->setLocalCOM(Eigen::Vector3d(xi, yi, zi - perturbedValue));
        revPertRobotArray[(i-1)*bodyParams + 4]->getBodyNode(i)->setMomentOfInertia(ixx - perturbedValue, iyy, izz, ixy, ixz, iyz);
        revPertRobotArray[(i-1)*bodyParams + 5]->getBodyNode(i)->setMomentOfInertia(ixx, iyy - perturbedValue, izz, ixy, ixz, iyz);
        revPertRobotArray[(i-1)*bodyParams + 6]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz - perturbedValue, ixy, ixz, iyz);
        revPertRobotArray[(i-1)*bodyParams + 7]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz, ixy - perturbedValue, ixz, iyz);
        revPertRobotArray[(i-1)*bodyParams + 8]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz, ixy, ixz - perturbedValue, iyz);
        revPertRobotArray[(i-1)*bodyParams + 9]->getBodyNode(i)->setMomentOfInertia(ixx, iyy, izz, ixy, ixz, iyz - perturbedValue);  
    }

/*============================================================================================*/
/*=======================================Calculate Phi========================================*/
/*============================================================================================*/

    cout << "|-> Done\n";
    cout << "Calculating Phi Matrix ...\n";

    ofstream dataTorque;
    dataTorque.open ("dataTorque_RHS.txt");
    dataTorque<< "dataTorque" << endl;

    ofstream phibetaRHS;
    phibetaRHS.open ("phibeta-RHS");
    phibetaRHS<< "phibeta-RHS" << endl;

    // ofstream RHSperturb;
    // RHSperturb.open ("RHS_perturb.txt");
    // RHSperturb<< "RHS_perturb" << endl;

    ofstream phifile;
    phifile.open ("phi.txt");

    // Eigen::MatrixXd phi_Mat(numDataPts*(17), numBetaVals);
    Eigen::MatrixXd phiMatrix(numBodies-1, numBetaVals);
    Eigen::MatrixXd phi(numBodies-1,1);
	
	for (int i = 0; i < numDataPts; i++) { //for each data point
		// Set idealRobot to compare torques with phi*beta calculation
		idealRobot->setPositions(allInitq.row(i));
        idealRobot->setVelocities(allInitqdot.row(i));
        Eigen::VectorXd ddq = allInitqdotdot.row(i);
        Eigen::MatrixXd M = idealRobot->getMassMatrix(); // n x n
        Eigen::VectorXd C = idealRobot->getCoriolisForces(); // n x 1
        Eigen::VectorXd G = idealRobot->getGravityForces(); // n x 1
        Eigen::VectorXd RHS_ideal= M*ddq + C + G;
        dataTorque << RHS_ideal.transpose() << endl;

        // For each Robot (each one has one perturbed value)
        for (int k = 0; k < numBetaVals; k++) { //for 170 loops

			// Set foward perturbed Robot
            fwdPertRobotArray[k]->setPositions(allInitq.row(i));
            fwdPertRobotArray[k]->setVelocities(allInitqdot.row(i));
            Eigen::MatrixXd M_pertfwd = fwdPertRobotArray[k]->getMassMatrix(); // n x n
            Eigen::VectorXd C_pertfwd = fwdPertRobotArray[k]->getCoriolisForces(); // n x 1
            Eigen::VectorXd G_pertfwd = fwdPertRobotArray[k]->getGravityForces(); // n x 1
            Eigen::VectorXd RHS_pertfwd = M_pertfwd*ddq + C_pertfwd + G_pertfwd; //}

            // Set reverse perturbed Robot
            revPertRobotArray[k]->setPositions(allInitq.row(i));
            revPertRobotArray[k]->setVelocities(allInitqdot.row(i));
            Eigen::MatrixXd M_pertrev = revPertRobotArray[k]->getMassMatrix(); // n x n
            Eigen::VectorXd C_pertrev = revPertRobotArray[k]->getCoriolisForces(); // n x 1
            Eigen::VectorXd G_pertrev = revPertRobotArray[k]->getGravityForces(); // n x 1
            Eigen::VectorXd RHS_pertrev = M_pertrev*ddq + C_pertrev + G_pertrev;

            // Calculate phi for beta i and pose
            phi = (RHS_pertfwd - RHS_pertrev)/(2*perturbedValue);
            // Add phi to phiMatrix and then print it looks cleaner
            phiMatrix.col(k) = phi;
        }

		// Fix phi
		for(int b=1; b<numBodies; b++) { //for 17 loops
			int c = bodyParams*(b-1); //0 10 20 ... 170
			double m = idealRobot->getBodyNode(b)->getMass();
			Eigen::Vector3d COM = idealRobot->getBodyNode(b)->getLocalCOM();

			phiMatrix.block<17,3>(0,c+1) = phiMatrix.block<17,3>(0,c+1)/m;
            phiMatrix.col(c) = phiMatrix.col(c) - phiMatrix.col(c+1)*COM(0) - phiMatrix.col(c+2)*COM(1) - phiMatrix.col(c+3)*COM(2);
		}
	
		Eigen::MatrixXd rhs_phibeta_diff(17,3);
		rhs_phibeta_diff <<  RHS_ideal, (phiMatrix*betaParams.transpose()), ((phiMatrix*betaParams.transpose()) - RHS_ideal);
		
		phibetaRHS<< "RHS, phi*beta, difference at "<< i << endl << endl << rhs_phibeta_diff << endl << endl;
		phibetaRHS<< "=========================================================================" << endl << endl << endl << endl;
		phifile<< phiMatrix.block<1,170>(0,0) << endl << endl << endl;
		// RHSperturb<<"PHI_MAT AT:" << i << endl << endl;
		// RHSperturb<< phiMatrix << endl;
	}
	dataTorque.close();
	phibetaRHS.close();
	// RHSperturb.close();
	phifile.close();
}

Eigen::MatrixXd readInputFileAsMatrix(string inputFilename) {
    ifstream infile;
    infile.open(inputFilename);

    if (!infile.is_open()) {
        throw runtime_error(inputFilename + " can not be read, potentially does not exit!");
    }

    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    while(! infile.eof()) {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];
        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();
    rows--;

    numDataPts = rows;

    // Populate matrix with numbers.
    Eigen::MatrixXd outputMatrix(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            outputMatrix(i,j) = buff[cols*i+j];

    return outputMatrix;
}

int main() {
    genPhiMatrixAsFile();
}