// Author: Akash Patel (apatel435@gatech.edu), Areeb Mehmood (amehmood7@gatech.edu)

// genPhiMatrixAsFile
// Purpose: Determine phi vectors for each input pose
// This phi will be used for finding beta (weights of actual robot) via gradient descent or some other method
//
// Input: Ideal beta={mi, MXi, MYi, ...}, krang urdf model, perturbation value,
// potentially unbalanced data points (q/poses) as a file
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
    string inputQFilename = "../../24-ParametricIdentification-Waist/simOutData/qWaistData.txt";
    string inputQdotFilename = "../../24-ParametricIdentification-Waist/simOutData/dqWaistData.txt";
    string inputQdotdotFilename = "../../24-ParametricIdentification-Waist/simOutData/ddqWaistData.txt";
    string inputTorqueFilename = "../../24-ParametricIdentification-Waist/simOutData/torqueWaistData.txt";
    string inputMassMatrixFilename = "../../24-ParametricIdentification-Waist/simOutData/mWaistData.txt";
    string inputCgFilename = "../../24-ParametricIdentification-Waist/simOutData/cgWaistData.txt";

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
    dart::dynamics::SkeletonPtr idealRobot = loader.parseSkeleton("/home/krang/dart/09-URDF/KrangWaist/krang_fixed_base.urdf");
    idealRobot->setGravity(Eigen::Vector3d (0.0, -9.81, 0.0));
    
    // Get ideal beta
    // Beta Definition/Format
    // mi, mxi, myi, mzi for each body

    int bodyParams = 10;
    int numBodies = idealRobot->getNumBodyNodes(); //returns 18
    dart::dynamics::BodyNodePtr bodyi;
    string namei;
    // These are our 10 parameters, + 3 variables xi, yi, zi
    double mi;
    double xi, xMi;
    double yi, yMi;
    double zi, zMi;
    double ixx, ixy, ixz, iyy, iyz, izz;

    int numBetaVals = (numBodies-1)*bodyParams;
    Eigen::MatrixXd betaParams(1, numBetaVals);

	// Fill betaParams array with values from URDF (masses, inertias) 
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

    cout << "|-> Done\n";

    // Save beta parameters
    ofstream betafile;
    betafile.open ("betaparameters.txt");
    betafile<< betaParams.transpose()<<endl;
    betafile.close();

    // Rotor Gear Ratios
    Eigen::VectorXd G_R(17);
    Eigen::VectorXd km(17);

    G_R(0)=596*2;   km(0) = 31.4e-3*2;	// Waist Motors 1 and 2
    G_R(1)=596;     km(1) = 31.4e-3;	//Torso
    G_R(2)=0;       km(2) = 0;			// Kinect N/A

    G_R(3)=596;		km(3) = 31.4e-3;	// Left Arm
    G_R(4)=596;		km(4) = 31.4e-3;
    G_R(5)=625;		km(5) = 38e-3;
    G_R(6)=625;		km(6) = 38e-3;
    G_R(7)=552;		km(7) = 16e-3;
    G_R(8)=552;		km(8) = 16e-3;
    G_R(9)=552;		km(9) = 16e-3;

    G_R(10)=596;    km(10) = 31.4e-3;	//Right Arm
    G_R(11)=596;	km(11) = 31.4e-3;
    G_R(12)=625;	km(12) = 38e-3;
    G_R(13)=625;	km(13) = 38e-3;
    G_R(14)=552;	km(14) = 16e-3;
    G_R(15)=552;	km(15) = 16e-3;
    G_R(16)=552;	km(16) = 16e-3;

    /*============================================================================================*/
    /*====================================Load array of Robot=====================================*/
    /*============================================================================================*/

    cout << "Creating robot array ...\n";
    // Load robots into fwdPertRobotArray and revPertRobotArray
    dart::dynamics::SkeletonPtr fwdPertRobotArray[numBetaVals];
    dart::dynamics::SkeletonPtr revPertRobotArray[numBetaVals];
    for(int i=0; i<numBetaVals; i++) {
        fwdPertRobotArray[i] = idealRobot->clone();
        revPertRobotArray[i] = idealRobot->clone();
    }

    // Perturb all Beta values in the forward direction
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

	// Perturb all Beta values in the reverse direction
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
    dataTorque.open ("../../24-ParametricIdentification-Waist/phiData/dataTorque_RHS.txt");
    dataTorque<< "dataTorque" << endl;

    ofstream phibetaRHS;
    phibetaRHS.open ("../../24-ParametricIdentification-Waist/phiData/phibeta-RHS");
    phibetaRHS<< "phibeta-RHS" << endl;

    ofstream phifile;
    phifile.open ("../../24-ParametricIdentification-Waist/phiData/phi.txt");

    ofstream phifile_comp;
    phifile_comp.open ("../../24-ParametricIdentification-Waist/phiData/phicomp.txt");

    Eigen::MatrixXd phiMatrix(numBodies-1, numBetaVals);
	Eigen::MatrixXd phiMatrix_comp(numBodies-1,numBetaVals+17*3); 
    Eigen::MatrixXd phi(numBodies-1,1);
	
	Eigen::MatrixXd gear_mat    = Eigen::MatrixXd::Zero(17,17);
	Eigen::MatrixXd viscous_mat = Eigen::MatrixXd::Zero(17,17);
	Eigen::MatrixXd coulomb_mat = Eigen::MatrixXd::Zero(17,17);
	
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

        for (int j=0;j<17;j++){
            gear_mat(j,j)    =  G_R(j)*G_R(j)*ddq(j);
            viscous_mat(j,j) =  allInitqdot.row(i)(j);
            coulomb_mat(j,j) = sin(allInitqdot.row(i)(j));
        }

        cout << i << " " << viscous_mat << endl;
	
        Eigen::MatrixXd rhs_phibeta_diff(17,3);
        rhs_phibeta_diff <<  RHS_ideal, (phiMatrix*betaParams.transpose()), ((phiMatrix*betaParams.transpose()) - RHS_ideal);
		
        phibetaRHS<< "RHS, phi*beta, difference at "<< i << endl << endl << rhs_phibeta_diff << endl << endl;
        phibetaRHS<< "=========================================================================" << endl << endl << endl << endl;
        phifile<< phiMatrix.block<1,170>(0,0) << endl << endl << endl;
		
		for(int j=0; j<17; j++){
			phiMatrix_comp.block<17,10>(0,j*13) = phiMatrix.block<17,10>(0,j*10);
			phiMatrix_comp.col(13*(j+1) - 3) = gear_mat.col(j);
			phiMatrix_comp.col(13*(j+1) - 2) = viscous_mat.col(j);
			phiMatrix_comp.col(13*(j+1) - 1) = coulomb_mat.col(j);
            // cout << phiMatrix_comp.col(13*(j+1)) << endl;
		}
        phifile_comp<< phiMatrix_comp;
    }
    dataTorque.close();
    phibetaRHS.close();
    phifile.close();
    phifile_comp.close();
}

// Read in files
Eigen::MatrixXd readInputFileAsMatrix(string inputFilename) {
    ifstream infile;
    infile.open(inputFilename);

    if (!infile.is_open()) {
        throw runtime_error(inputFilename + " can not be read, potentially does not exist!");
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