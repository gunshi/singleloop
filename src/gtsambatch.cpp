#include "includes/gtsambatch.h"
#include <gtsam/slam/InitializePose3.h>

using namespace std;
using namespace gtsam;

void batch3d(string filename){
      
  std::pair<NonlinearFactorGraph::shared_ptr, Values::shared_ptr> data = readG2o(filename,true);
  
  Values initial(*data.second);
  // Add prior on the first key
  NonlinearFactorGraph dataset=*data.first;
    NonlinearFactorGraph graphWithPrior = *data.first;
    gtsam::Vector v(6);
    v<<1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    noiseModel::Diagonal::shared_ptr priorModel = noiseModel::Diagonal::Variances(v);
    Key firstKey = 1;

      graphWithPrior.add(PriorFactor<Pose3>(firstKey, Pose3(), priorModel));


    std::cout << "Initializing Pose3 - chordal relaxation" << std::endl;
    Values initialization = InitializePose3::initialize(graphWithPrior);
    std::cout << "done!" << std::endl;

  std::cout<<"values, dataset sizes : "<<initialization.size()<<", " <<graphWithPrior.size()<<endl;
  LevenbergMarquardtParams lmparams;
  LevenbergMarquardtOptimizer lmoptimizer(dataset,initialization,lmparams);
  Values result;
  result=lmoptimizer.optimize();
  const string outputfilelm="encounters.g2o";
  writeG2o(dataset,result,outputfilelm);
    
}
