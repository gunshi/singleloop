// To find number of images in a directory

#include <sys/types.h>
#include <dirent.h>
#include "helperfunctions.h"
#include "filewriter.h"
#include "threadcomm.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdint.h>
#include <sstream> 

#include <viso_stereo.h>
//#include <png++/png.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>

// g2o optimization
#include <g2o/core/sparse_optimizer.h>
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include "g2o/core/factory.h"
#include "g2o/stuff/command_args.h"

using namespace std;
using namespace cv;
using namespace g2o;

// we use the 2D and 3D SLAM types here
G2O_USE_TYPE_GROUP(slam2d)
G2O_USE_TYPE_GROUP(slam3d)

#define PI 3.14159265

writer mywriter;
comm mythread_comm;

pthread_mutex_t myMutex;

double valx;
double valz;

int listdir(char *dir) {
  struct dirent *dp;
  DIR *fd;
  int counter =0;
  //cout << dir<<endl;

  if ((fd = opendir(dir)) == NULL) {
    fprintf(stderr, "listdir: can't open %s\n", dir);
    return -1;
  }
  while ((dp = readdir(fd)) != NULL) {
  if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, ".."))
    continue;    /* skip self and parent */
  // printf("%s/%s\n", dir, dp->d_name);
  	counter++;
  //printf("%s\n", dp->d_name);
  }
  //cout<< counter<<endl;
  closedir(fd);
  return counter;
}

struct for_g2o_thread
{
  string s1;
  //string outfile;
  int maxIterations;
};

 void *g2o_thread(void *t)
 {
  cout << "entering g2o thread!" << endl;
  for_g2o_thread* obj = (for_g2o_thread *) t;
  stringstream mystring;
  
  //cout << "string" << obj->s1 << endl;
  mystring << obj->s1;
  //cout << "mystring " << mystring.str() << endl;

   // create the linear solver
  BlockSolverX::LinearSolverType * linearSolver = new LinearSolverCSparse<BlockSolverX::PoseMatrixType>();

  // create the block solver on top of the linear solver
  BlockSolverX* blockSolver = new BlockSolverX(linearSolver);

  // create the algorithm to carry out the optimization
  //OptimizationAlgorithmGaussNewton* optimizationAlgorithm = new OptimizationAlgorithmGaussNewton(blockSolver);
  OptimizationAlgorithmLevenberg* optimizationAlgorithm = new OptimizationAlgorithmLevenberg(blockSolver);

  // NOTE: We skip to fix a variable here, either this is stored in the file
  // itself or Levenberg will handle it.

  // create the optimizer to load the data and carry out the optimization
  SparseOptimizer optimizer;
  optimizer.setVerbose(true);
  optimizer.setAlgorithm(optimizationAlgorithm);

  //cout << mystring.str() << endl;

  optimizer.load1(mystring);  

  optimizer.initializeOptimization();
  optimizer.optimize(obj->maxIterations);

  //cout << "saving g2o file" << obj->outfile << " ... ";

  string out = optimizer.save1(obj->s1,0);
  obj->s1 = "";
  mywriter.g2o_string = "";
  mywriter.g2o_string = out;

  cout << "exiting g2o thread!!" << endl;

  return(void*)obj;
 }



void my_libviso2(std::vector<Matrix> &Tr_local, std::vector<Matrix> &Tr_global, string dir, int numImg,gtsam::ISAM2 &isam2,gtsam::NonlinearFactorGraph &nfg) {

  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereo::parameters param;
  
  // calibration parameters for sequence 2010_03_09_drive_0019 
/**  
param.calib.f  = 711.9212646484375; // focal length in pixels
  param.calib.cu = 647.0408325195312; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.7899169921875; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters
**/

 // calibration parameters for sequence 2010_03_09_drive_0019
  param.calib.f  = 722.658813477; // focal length in pixels
  param.calib.cu = 619.283325195; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.0; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters

  pthread_mutex_init(&myMutex,0);
  pthread_t g2othread;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for_g2o_thread gthread;
  //gthread.outfile = outputFilename;
  gthread.maxIterations = 10;

  mywriter.my_write_file.open("data/offline.g2o");
  string ss;
  
  mythread_comm.loop_wait = true;
  
  // init visual odometry
  VisualOdometryStereo viso(param);
  
  Matrix pose = Matrix::eye(4);
  Tr_global.push_back(pose);
  
  Matrix extra_relations;
  bool is_good = false;

  for (int32_t i=1; i<=numImg; i++) 
  {  

  if(i==1){
    //init isam2 with prior
    gtsam::NonlinearFactorGraph newFactors;
    gtsam::Values newVariables;
    newFactors.push_back(boost::make_shared<gtsam::PriorFactor<gtsam::Pose3> >(1, gtsam::Pose3(), gtsam::noiseModel::Unit::Create(gtsam::Pose3::dimension)));
    newVariables.insert(1, gtsam::Pose3()); //are we sure about 1?
    gtsam::ISAM2Result res =  isam2.update(newFactors, newVariables); //need to store result?
  }

  pthread_mutex_lock(&myMutex);
  mythread_comm.viso_wait_flag = true;
  pthread_mutex_unlock(&myMutex);

  // input file names
  char base_name[256]; sprintf(base_name,"%04d.jpg",i);
  string left_img_file_name  = dir + "left/" + base_name;
  string right_img_file_name = dir + "right/" + base_name;
  
  // catch image read/write errors here
  try 
  {

    // load left and right input image
    Mat left_img_src, right_img_src;
    left_img_src = imread(left_img_file_name,CV_LOAD_IMAGE_COLOR);
    right_img_src = imread(right_img_file_name,CV_LOAD_IMAGE_COLOR);

    Mat left_img,right_img;
    cvtColor(left_img_src,left_img,CV_BGR2GRAY);
    cvtColor(right_img_src,right_img,CV_BGR2GRAY);
    
    int32_t width  = left_img.cols;
    int32_t height = left_img.rows;

    // convert input images to uint8_t buffer
    uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    int32_t k=0;
    for (int32_t row=0; row < left_img.rows; row++) {
      for (int32_t col=0; col < left_img.cols; col++) {
        left_img_data[k]  = left_img.at<uchar>(row,col);
        right_img_data[k] = right_img.at<uchar>(row,col);
        k++;
      }
    }

    // status
    cout << "Processing: Frame: " << i << endl;
    
    // compute visual odometry
    int32_t dims[] = {width,height,width};
    
      viso.process(left_img_data,right_img_data,dims);
      // on success, update current pose

      //Tr_local.push_back(Matrix::inv(viso.getMotion()));
      
      
      if(i%10==0) //try without this also
      {
        pthread_mutex_lock(&myMutex);
        while(mythread_comm.isamupdating==true){}
        mythread_comm.isamupdating=true;
        ss = my_for_g2o_edge(i,i+1,Matrix::inv(viso.getMotion()),isam2,true,nfg);
        mythread_comm.isamupdating=false;
        pthread_mutex_unlock(&myMutex);
      }
      else{
          pthread_mutex_lock(&myMutex);
          while(mythread_comm.isamupdating==true){}

          mythread_comm.isamupdating=true;

        ss = my_for_g2o_edge(i,i+1,Matrix::inv(viso.getMotion()),isam2,true,nfg);
        mythread_comm.isamupdating=false;

          pthread_mutex_unlock(&myMutex);
      }
      mywriter.my_write_file << ss;
      mywriter.g2o_string = mywriter.g2o_string + ss;

      if((i%20==0) && (i+5< numImg))
      {
        is_good = my_libviso2_relative(extra_relations,i,i+5,dir);

          if(is_good)
          {
            pthread_mutex_lock(&myMutex);
            ss = my_for_g2o_edge(i,i+5,extra_relations,isam2,true,nfg);
            pthread_mutex_unlock(&myMutex);

            mywriter.my_write_file << ss;
            //g2o_string << ss;
            mywriter.g2o_string = mywriter.g2o_string + ss;
          }

          cout << "extra!! " << i << " " << i+5 << endl;
      }

      if(i>1)
      {
        pose = pose * Matrix::inv(viso.getMotion());
       // Tr_global.push_back(pose);  
      }
      

      // // output some statistics
      // double num_matches = viso.getNumberOfMatches();
      // double num_inliers = viso.getNumberOfInliers();
      // cout << ", Matches: " << num_matches;
      // cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
      // cout << pose << endl << endl;

    // release uint8_t buffers
    free(left_img_data);
    free(right_img_data);

  // catch image read errors here
  } catch (...) {
    cerr << "ERROR: Couldn't read input files!" << endl;
    break;
  } 
  while(mythread_comm.loop_wait==true)
  {

  }

//  //needed?
//  if(((i%500==0) || i==numImg) && mythread_comm.g2o_loop_flag==true) 
//  {
//       
//    //cout << "in function " << gthread.s1 << endl;
//    gthread.s1 =  mywriter.g2o_string + "\nFIX 1\n";;
//    pthread_create(&g2othread, &attr, g2o_thread, (void *)&gthread);
//    pthread_join(g2othread,NULL);
//    
//    pthread_mutex_lock(&myMutex);
//    mythread_comm.g2o_loop_flag = false;
//    pthread_mutex_unlock(&myMutex);
//
//  }
  pthread_mutex_lock(&myMutex);
  mythread_comm.viso_wait_flag = false;
  pthread_mutex_unlock(&myMutex);
 }

pthread_mutex_destroy(&myMutex);

 while(!mythread_comm.loop_write_done)
  {}


ofstream g2o_file;
g2o_file.open("data/online.g2o");

g2o_file << mywriter.g2o_string << "\nFIX 1\n";
g2o_file.close();

mywriter.my_write_file << "\nFIX 1 \n";
mywriter.my_write_file.close();

const string outputfile="isam.g2o";
gtsam::Values estimate(isam2.calculateEstimate());
gtsam::NonlinearFactorGraph datasetempty;
gtsam::writeG2o(datasetempty,estimate,outputfile);
//write nfg also? !!
}


bool my_libviso2_relative(Matrix &Tr_final, int index1, int index2, std::string dir)
{
   // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereo::parameters param;
  
  
  // calibration parameters for sequence 2010_03_09_drive_0019 
/**  
param.calib.f  = 711.9212646484375; // focal length in pixels
  param.calib.cu = 647.0408325195312; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.7899169921875; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters
**/

  param.calib.f  = 722.658813477; // focal length in pixels
  param.calib.cu = 619.283325195; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.0; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters

  // init visual odometry
  VisualOdometryStereo viso(param);
  
  // current pose (this matrix transforms a point from the current

    Matrix pose = Matrix::eye(4);
   // Tr_relative r;
    string left_img_file_name;
    string right_img_file_name;

    for (int j = 1; j < 3; j++)
    {
      if (j==1)
      {
        char base_name[256]; sprintf(base_name,"%04d.jpg",index1);
        left_img_file_name  = dir + "left/" + base_name;
        right_img_file_name = dir + "right/" + base_name;
       // r.frame1 = index1;
      }
      else
      {
        char base_name[256]; sprintf(base_name,"%04d.jpg",index2);
        left_img_file_name  = dir + "left/" + base_name;
        right_img_file_name = dir + "right/" + base_name; 
       // r.frame2 = index2;
      }
        
      try {

      // load left and right input image
      Mat left_img_src, right_img_src;
      left_img_src = imread(left_img_file_name,CV_LOAD_IMAGE_COLOR);
      right_img_src = imread(right_img_file_name,CV_LOAD_IMAGE_COLOR);

      Mat left_img,right_img;
      cvtColor(left_img_src,left_img,CV_BGR2GRAY);
      cvtColor(right_img_src,right_img,CV_BGR2GRAY);
      // png::image< png::gray_pixel > left_img(left_img_file_name);
      // png::image< png::gray_pixel > right_img(right_img_file_name);

      // image dimensions
      // int32_t width  = left_img.get_width();
      // int32_t height = left_img.get_height();

      int32_t width  = left_img.cols;
      int32_t height = left_img.rows;

      // convert input images to uint8_t buffer
      uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      int32_t k=0;
        for (int32_t row=0; row < left_img.rows; row++) {
          for (int32_t col=0; col < left_img.cols; col++) {
          // left_img_data[k]  = left_img.get_pixel(u,v);
          // right_img_data[k] = right_img.get_pixel(u,v);
            left_img_data[k]  = left_img.at<uchar>(row,col);
            right_img_data[k] = right_img.at<uchar>(row,col);
            k++;
          }
        }

      // status
        cout << "Processing: Frame between: " << index1 << '\t' << index2 << endl;
      
        // compute visual odometry
        int32_t dims[] = {width,height,width};
      
      
        bool pass = viso.process(left_img_data,right_img_data,dims);
        if(pass){
            cout<<"j="<<j<<endl;
            cout<<"passed"<<endl;

        }
        else{
            cout<<"j="<<j<<endl;
            cout<<"failed"<<endl;

        }
        // on success, update current pose

        // output some statistics

        double num_matches = viso.getNumberOfMatches();
        double num_inliers = viso.getNumberOfInliers();
        double inliers_percent = 100.0*num_inliers/num_matches;
        cout << ", Matches: " << num_matches << endl;
        cout << ", Inliers: " << inliers_percent << " %" << endl;
        // cout << pose << endl << endl;
         if(j>1 && inliers_percent>=10 && pass )
        {
          pose = pose * Matrix::inv(viso.getMotion());
          //r.transform = pose;
          Tr_final = pose;  
          
          free(left_img_data);
          free(right_img_data);

          return true;
          //cout << "i: " << i << '\t' << "j: " << j <<endl << endl;
        }

      // release uint8_t buffers
      free(left_img_data);
      free(right_img_data);

      // catch image read errors here
    } 
      catch (...) 
      {
        cerr << "ERROR: Couldn't read input files!" << endl;
        break;
      }
   }

   return false;
}


string my_for_g2o_edge(int id1, int id2, Matrix transform, gtsam::ISAM2 &isam2,bool relin,gtsam::NonlinearFactorGraph &nfg)
{

  stringstream out;
  Matrix new_tr; 
  new_tr = transform;
  
  double dx = new_tr.getData(0,3);
  double dy = new_tr.getData(1,3);
  double dz = new_tr.getData(2,3);

  double thetay = atan2(new_tr.getData(0,2), sqrt(new_tr.getData(0,0)*new_tr.getData(0,0) + new_tr.getData(0,1)*new_tr.getData(0,1)));

  double thetax = atan2(-new_tr.getData(1,2), new_tr.getData(2,2));

  double thetaz = atan2(-new_tr.getData(0,1), new_tr.getData(0,0));
    out << "EDGE3 " << id1 <<' '<< id2 <<' '<< dx << ' '<< dy <<' '<< dz <<' '<< thetax << ' ' << thetay <<' '<< thetaz<<" "<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
  
    
    
  gtsam::Matrix M(6, 6);
  if(abs(id1-id2)>5)
  	{ M << 10000,0,0,0,0,0,0,10000,0,0,0,0,0,0,10000,0,0,0,0,0,0,10000,0,0,0,0,0,0,10000,0,0,0,0,0,0,10000;}
  else
  	{ M << 1000,0,0,0,0,0,0,1000,0,0,0,0,0,0,1000,0,0,0,0,0,0,1000,0,0,0,0,0,0,1000,0,0,0,0,0,0,1000;}
  gtsam::SharedNoiseModel model;
  model = gtsam::noiseModel::Gaussian::Information(M, true);
  
      gtsam::Values newVariables;
      gtsam::NonlinearFactorGraph newFactors;
      gtsam::Rot3 rot1(new_tr.getData(0,0),new_tr.getData(0,1),new_tr.getData(0,2),new_tr.getData(1,0),new_tr.getData(1,1),new_tr.getData(1,2),new_tr.getData(2,0),new_tr.getData(2,1),new_tr.getData(2,2));
      gtsam::Point3 point(dx,dy,dz);
      gtsam::Pose3 l1Xl2(rot1,point);
      
      gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(id1, id2, l1Xl2, model));
      factor->print();
      nfg.push_back(factor);
        if(gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr observation = 
            boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor))
        {
                newFactors.push_back(observation);
                if(observation->key1() > observation->key2()){
                  if(!isam2.valueExists(observation->key1()) ){                     
                    if(!isam2.valueExists(observation->key2())){
                        throw runtime_error("Problem! None exist");
                    }
                    if(id1 == 1){
                      newVariables.insert(observation->key1(), observation->measured().inverse());
                    }
                    else{
                      gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key2());
                      newVariables.insert(observation->key1(), previousPose * observation->measured().inverse());
                    }
                  }
                }
                else {  //+5 frame constraint case
                  if(!isam2.valueExists(observation->key2()) ){
                    if(!isam2.valueExists(observation->key1())){
                        throw runtime_error("Problem! None exist");
                    }
                    if (id1 == 1){
                      newVariables.insert(observation->key2(), observation->measured());
                    }
                    else{
                      gtsam::Pose3 previousPose = isam2.calculateEstimate<gtsam::Pose3>(observation->key1());
                      newVariables.insert(observation->key2(), previousPose * observation->measured());
                    }
                  }
                }
                      
        }
     
     
	if(relin){
            std::cout<<"ASKED TO RELIN"<<std::endl;
        gtsam::ISAM2Result res1 =  isam2.update(newFactors, newVariables,gtsam::FactorIndices(),boost::none,boost::none,boost::none, true);
        std::cout<< "variables relinearised (non zero)= "<<res1.variablesRelinearized<<std::endl;
	}
	else{
	gtsam::ISAM2Result res1 = isam2.update(newFactors,newVariables);
        std::cout<< "variables relinearised (should be zero)= "<<res1.variablesRelinearized<<std::endl;
	}

  return out.str();
}
