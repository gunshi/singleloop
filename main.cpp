#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "highgui.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <pthread.h>

#include "includes/helperfunctions.h"
#include "includes/gtsambatch.h"
// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines Surf64Vocabulary
#include "DLoopDetector.h" // defines Surf64LoopDetector
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX
#include <DUtils/DUtils.h>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
// Demo
#include "includes/demoDetector.h"

#include <DVision/DVision.h>
//using namespace DVision;
//surfvocab where defined


using namespace DLoopDetector;
using namespace DBoW2;
using namespace DUtils;
using namespace std;
using namespace cv;

#define PI 3.14159265

// Directroy of Images
string IMG_DIR1 = "/home/gunshi/Downloads/exp_12_10_2_sync_imgs/";
//string IMG_DIR1 = "/home/gunshi/Downloads/newrun/loop3/";
//string IMG_DIR1 = "/media/root/Heavy_Dataset/newrun/sampled/loop2/";
//string IMG_DIR1 = "/home/tushar/Datasets/Datasets/cair_data/loop1/";
// DLoop resources
static const string VOC_FILE = "./resources/iiit2_voc.voc.gz";
static const string IMAGE_DIR = IMG_DIR1 + "left/";
static const int IMAGE_W = 640; // image size
static const int IMAGE_H = 480;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
void my_dloop(std::vector<Matrix> &m,std::vector<int> &index1, std::vector<int> &index2,string file, string img_dir, int a, int b, gtsam::ISAM2 &isam2,gtsam::NonlinearFactorGraph &nfg);
void loadFeaturess(vector<vector<vector<float> > > &features);
void changeStructure(const vector<float> &plain, vector<vector<float> > &out,int L);
void testVocCreations(const vector<vector<vector<float> > > &features);
void testDatabases(const vector<vector<vector<float> > > &features);


/// This functor extracts SURF64 descriptors in the required format


class SurfExtractor: public FeatureExtractor<FSurf64::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const;
};

// Struct for passing variables to viso thread
struct for_libviso_thread
{
  std::vector<Matrix> Tr_local;
  std::vector<Matrix> Tr_global;
  int numimages;
  string imgdir;
  gtsam::ISAM2 *isam2;
  gtsam::NonlinearFactorGraph *nfg;
};

// Struct for passing variables to Dloop thread
struct for_dloop_thread
{
  std::vector<Matrix> mat;
  std::vector<int> i1;
  std::vector<int> i2;
  gtsam::ISAM2 *isam2;
  gtsam::NonlinearFactorGraph *nfg;
};

// Viso Thread
 void *libviso_thread(void *t)
 {
    cout << "entering libviso2 thread!" << endl;
    for_libviso_thread* obj = (for_libviso_thread *) t;

    // Calling libviso function
    my_libviso2(obj->Tr_local,obj->Tr_global,obj->imgdir,obj->numimages,*(obj->isam2),*(obj->nfg));
    
    cout << "exiting libviso thread!!" << endl;
    return (void *)obj;
 }

// DLoop Thread
 void *dloop_thread(void *t)
{
    cout << "entered Dloop thread! "<< endl;
    for_dloop_thread* obj = (for_dloop_thread *) t;

    // Calling Dloop function
    my_dloop(obj->mat,obj->i1,obj->i2,VOC_FILE,IMG_DIR1,IMAGE_W,IMAGE_H,*(obj->isam2),*(obj->nfg));

    cout << "exiting Dloop thread!!" << endl;
    return (void *)obj;
 }

/**
 void loadFeatures(vector<vector<FSurf64::TDescriptor > > &features)
 {
   int NIMAGES=1955; //1955,2097 loop2 loop3
   features.clear();
   features.reserve(NIMAGES);
   cout << "Extracting BRIEF features..." << endl;
   for(int i = 1; i <= NIMAGES; ++i)
   {
     stringstream ss;
     ss << "/media/root/Heavy_dataset/newrun/sampled/loop2/left/" <<  setfill('0') << setw(4) << i << ".jpg";
     cout << ss.str() << endl;
     cv::Mat image = cv::imread(ss.str(), 0);
     cv::Mat mask;
     vector<cv::KeyPoint> keypoints, kpt;
     //cv::Mat descriptors;
     vector<FSurf64::TDescriptor> descriptors_1;

     DVision::SurfSet m_brief;
     cv::FAST(image, kpt, 20, true);
     //descriptors_1 assign, descriptors
     //std::vector<float> descriptors; internal public memeber
     m_brief.Compute(image, kpt,false);
     features.push_back(descriptors_1);
   }
 }

// void loadBriefFeatures(vector<vector<BRIEF::bitset > > &features)
// {
  // int NIMAGES=1955;
  // features.clear();
  // features.reserve(NIMAGES);

  // cout << "Extracting BRIEF features..." << endl;




  // for(int i = 1; i <= NIMAGES; ++i)
  // {
    // stringstream ss;
     //     ss << "/media/root/Heavy_Dataset/newrun/sampled/loop2/left/" <<  setfill('0') << setw(4) << i << ".jpg";
    // cout << ss.str() << endl;
    // cv::Mat image = cv::imread(ss.str(), 0);
    // cv::Mat mask;
    // vector<cv::KeyPoint> keypoints, kpt;
    // cv::Mat descriptors;
    // vector<BRIEF::bitset> descriptors_1;
    // DVision::BRIEF m_brief;
    // cv::FAST(image, kpt, 20, true);
    // m_brief.compute(image, kpt, descriptors_1);
   //  features.push_back(descriptors_1);
 //  }
// }

//void testBriefVocCreation(const vector<vector<BRIEF::bitset > > &features)
// {
   // branching factor and depth levels
  // const int k = 10;
  // const int L = 6;
  // const WeightingType weight = TF_IDF;
  // const ScoringType score = L1_NORM;

  // BriefVocabulary voc(k, L, weight, score);

  // cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  // voc.create(features);
  // cout << "... done!" << endl;


  // cout << "Vocabulary information: " << endl
  // << voc << endl << endl;

   // save the vocabulary to disk
  // cout << endl << "Saving vocabulary..." << endl;
  // voc.save("Brief_k10L6_edited.voc.gz");
  // cout << "Done" << endl;
// }

// void testVocCreation(const vector<vector<FSurf64::TDescriptor > > &features)
// {
   // branching factor and depth levels
  // const int k = 10;
  // const int L = 6;
  // const WeightingType weight = TF_IDF;
 //  const ScoringType score = L1_NORM;

 //  Surf64Vocabulary voc(k, L, weight, score);

  // cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  // voc.create(features);
  // cout << "... done!" << endl;
  // cout << "Vocabulary information: " << endl
  // << voc << endl << endl;

   // save the vocabulary to disk
  // cout << endl << "Saving vocabulary..." << endl;
  // voc.save("Surf64_k10L6_edited.voc.gz");
  // cout << "Done" << endl;
// }



const int NIMAGES =1955 ;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
void wait()
{
      cout << endl << "Press enter to continue" << endl;
        getchar();
}


void loadFeaturess(vector<vector<vector<float> > > &features)
{
      features.clear();
        features.reserve(NIMAGES);

          cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);

            cout << "Extracting SURF features..." << endl;
              for(int i = 1; i <1955 ; ++i)
                    {
                            stringstream ss;
                             ss << "/media/Heavy_dataset/newrun/sampled/loop2/left/" <<  setfill('0') << setw(4) << i << ".jpg"; 

                                    cv::Mat image = cv::imread(ss.str(), 0);
                                        cv::Mat mask;
                                            vector<cv::KeyPoint> keypoints;
                                                vector<float> descriptors;

                                                    surf->detectAndCompute(image, mask, keypoints, descriptors);

                                                        features.push_back(vector<vector<float> >());
                                                            changeStructure(descriptors, features.back(), surf->descriptorSize());
                                                              }
}
void testVocCreations(const vector<vector<vector<float> > > &features)
{

      const int k = 9;
        const int L = 3;
          const WeightingType weight = TF_IDF;
            const ScoringType score = L1_NORM;

              Surf64Vocabulary voc(k, L, weight, score);

                cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
                  voc.create(features);
                    cout << "... done!" << endl;

                      cout << "Vocabulary information: " << endl
                            << voc << endl << endl;

                        cout << endl << "Saving vocabulary..." << endl;
                          voc.save("iiit_voc.voc.gz");
                            cout << "Done" << endl;
}

void testDatabases(const vector<vector<vector<float> > > &features)
{
      cout << "Creating a small database..." << endl;


        Surf64Vocabulary voc("iiit_voc.voc.gz");
          
          Surf64Database db(voc, false, 0); 

            for(int i = 0; i < NIMAGES; i++)
                  {
                          db.add(features[i]);
                            }

              cout << "... done!" << endl;

                cout << "Database information: " << endl << db << endl;


                  cout << "Querying the database: " << endl;

                    QueryResults ret;
                      for(int i = 0; i < NIMAGES; i++)
                            {
                                    db.query(features[i], ret, 4);

                                        cout << "Searching for Image " << i << ". " << ret << endl;
                                          }

                        cout << endl;

                          cout << "Saving database..." << endl;
                            db.save("iiit_db.voc.gz");
                              cout << "... done!" << endl;
                                

                                cout << "Retrieving database once again..." << endl;
                                  Surf64Database db2("iiit_db.voc.gz");
                                    cout << "... done! This is: " << endl << db2 << endl;
}
void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
          int L)
{
      out.resize(plain.size() / L);

        unsigned int j = 0;
          for(unsigned int i = 0; i < plain.size(); i += L, ++j)
                {
                        out[j].resize(L);
                            std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
                              }
}

**/


// For single loop
int main()
{	//string writefile="master-isamnfg.g2o";
	//batch3d(writefile);
	//return 0;
 // vector<vector<vector<float> > > features;
   // loadFeaturess(features);

     // testVocCreations(features);

       // wait();

         // testDatabases(features);

           // return 0;
//    std::vector<std::vector<FSurf64::TDescriptor > > features;
//    SurfExtractor extractor;
//    demoDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
//     demo(VOC_FILE,IMG_DIR1,IMAGE_W,IMAGE_H);
//    demo.runVocab("SURF64", extractor,features);
//    testVocCreation(features);

//    std::vector<std::vector<BRIEF::bitset > > bfeatures;
//    loadBriefFeatures(bfeatures);
//    testBriefVocCreation(bfeatures);
//    return 0;
 
gtsam::ISAM2Params params;
params.optimizationParams = gtsam::ISAM2DoglegParams();
params.relinearizeSkip = 1;
params.enablePartialRelinearizationCheck = true;
gtsam::ISAM2 isam2(params);
gtsam::NonlinearFactorGraph nfg;
  pthread_t thread[2];
  pthread_attr_t attr;

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // Number of images in given directory
  int numImages1 =0;
  std::string path1 = IMG_DIR1 + "/left/";
  char * dst1 = new char[path1.length() + 1];
  std::strcpy(dst1,path1.c_str());

  numImages1 = listdir(dst1);

  for_libviso_thread lib;
  for_dloop_thread dthread;

  lib.imgdir = IMG_DIR1;
  lib.numimages = numImages1;
  lib.isam2 = &isam2;
  lib.nfg = &nfg;

  dthread.isam2 = &isam2;
  dthread.nfg = &nfg;
  // Create Dloop and Viso Thread 
  pthread_create(&thread[0], &attr, dloop_thread, (void *)&dthread);
  pthread_create(&thread[1], &attr, libviso_thread, (void *)&lib);  
  
  // Wait for Completion
  pthread_join(thread[0],NULL);
  pthread_join(thread[1],NULL);

  gtsam::Values estimate(isam2.calculateEstimate());

  string optimisedfile="3doptimisedloop"+string(1,IMG_DIR1[IMG_DIR1.size()-2])+".g2o";
  gtsam::NonlinearFactorGraph opt;
  
  int numKeys=estimate.size();
  for(int k=1;k<=numKeys;k++){
      gtsam::Pose3 tempPo = estimate.at<gtsam::Pose3>(k);
      gtsam::Pose3 tempPo2 = estimate.at<gtsam::Pose3>((k%numKeys)+1);
      gtsam::Pose3 btw = tempPo.between(tempPo2);
      gtsam::Matrix Info = isam2.marginalCovariance((k%numKeys)+1).inverse();
    gtsam::SharedNoiseModel model;
    model = gtsam::noiseModel::Gaussian::Information(Info, true);

    gtsam::NonlinearFactor::shared_ptr factor(
          new gtsam::BetweenFactor<gtsam::Pose3>(k, (k%numKeys)+1, btw, model));
    opt.push_back(factor);
  }
    gtsam::writeG2o(opt,gtsam::Values(),optimisedfile);
  cout << "done" << endl;

  return 0;
}


// DLoop helper functions
// ----------------------------------------------------------------------------

void SurfExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const
{
  // extract surfs with opencv
   static cv::Ptr<cv::xfeatures2d::SURF> surf_detector = 
    cv::xfeatures2d::SURF::create(400);
  
  surf_detector->setExtended(false);
  
  keys.clear(); // opencv 2.4 does not clear the vector
  vector<float> plain;
  surf_detector->detectAndCompute(im, cv::Mat(), keys, plain);
  
  // change descriptor format
  const int L = surf_detector->descriptorSize();
  descriptors.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    descriptors[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
  }
}

// ----------------------------------------------------------------------------

void my_dloop(std::vector<Matrix> &Mat,std::vector<int> &index1, std::vector<int> &index2,string VOC_FILE1,string IMAGE_DIR1, int IMAGE_W1, int IMAGE_H1, gtsam::ISAM2 &isam2,gtsam::NonlinearFactorGraph &nfg)
{
   demoDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
    demo(VOC_FILE1, IMAGE_DIR1, IMAGE_W1, IMAGE_H1);

  try 
  {  
    // run the demo with the given functor to extract features
    SurfExtractor extractor;
    demo.run(Mat,index1,index2,"SURF64", extractor,isam2,nfg);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }
}

