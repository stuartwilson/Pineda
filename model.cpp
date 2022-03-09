#include "pineda.h"

#include <morph/HdfData.h>
#include <morph/Config.h>

int main(int argc, char **argv){

        if (argc < 4) { std::cerr << "\nUsage: ./build/model configfile logdir seed"; return -1; }

        srand(time(NULL)); // note may not be different for simultaneously launched progs
        int seed = rand();
        if(argc==5){
            seed = std::stoi(argv[4]);
        }

        morph::RandUniform<double, std::mt19937> _rng(seed);
        rng = &_rng;

        std::string paramsfile (argv[1]);
        morph::Config conf(paramsfile);
        if (!conf.ready) { std::cerr << "Error setting up JSON config: " << conf.emsg << std::endl; return 1; }

        std::string logpath = argv[2];
        std::ofstream logfile;
        morph::Tools::createDir (logpath);
        { std::stringstream ss; ss << logpath << "/log.txt"; logfile.open(ss.str());}
        logfile<<"Hello."<<std::endl;

        double dt = conf.getFloat("dt", 1.0);
        double tauX = conf.getFloat("tauX", 1.0);
        double tauY = conf.getFloat("tauY", 1.0);
        double tauW = conf.getFloat("tauW", 32.0);
        double weightMin = conf.getFloat("weightMin", -1.0);
        double weightMax = conf.getFloat("weightMax",  1.0);
        double divThresh = conf.getFloat("divThresh", 0.000001);
        int maxSteps = conf.getInt("maxSteps",400);
        int errorSamplePeriod = conf.getInt("errorSamplePeriod",1000);

        int T = conf.getInt("T",100000);
        const Json::Value pre = conf.getArray("pre");
        const Json::Value post = conf.getArray("post");

        if(pre.size()!=post.size()){
            std::cout<<"Pre and Post vectors different sizes"<<std::endl;
            return 0;
        }

        // determine number of network nodes
        int N = -1;
        for(int i=0;i<pre.size();i++){
            if(pre[i].asInt()>N){
                N = pre[i].asInt();
            }
            if(post[i].asInt()>N){
                N = post[i].asInt();
            }
        }
        N++;

        // identify input and output nodes
        std::vector<int> inID(0);
        const Json::Value I = conf.getArray ("inputNodes");
        for (int i=0; i<I.size(); i++) {
            inID.push_back(I[i].asInt());
        }
        std::vector<int> ouID(0);
        const Json::Value O = conf.getArray ("outputNodes");
        for (int i=0; i<O.size(); i++) {
            ouID.push_back(O[i].asInt());
        }

        // read in map information
        std::string mapFileName = conf.getString("mapFileName", "unknown map");
        std::vector<double> In, Out;
        morph::HdfData map(mapFileName,1);
        map.read_contained_vals ("In", In);
        map.read_contained_vals ("Out", Out);

        int nPoint = In.size()/inID.size();
        if(nPoint !=Out.size()/ouID.size()){
            std::cout<<"Map input/output dims don't match."<<std::endl;
            return 0;
        }
        std::vector<std::vector<double> > inVals(nPoint,std::vector<double>(inID.size(),0.0));
        std::vector<std::vector<double> > ouVals(nPoint,std::vector<double>(ouID.size(),0.0));
        {
            int k=0;
            for(int i=0;i<nPoint;i++){
                for(int j=0;j<inID.size();j++){
                    inVals[i][j] = In[k];
                    k++;
                }
            }
        }
        {
            int k=0;
            for(int i=0;i<nPoint;i++){
                for(int j=0;j<ouID.size();j++){
                    ouVals[i][j] = Out[k];
                    k++;
                }
            }
        }

        // Create the network
        Network P(dt,N,tauX,tauY,tauW,divThresh,maxSteps);

        // Identify the input and output nodes
        P.setInputID(inID);
        P.setOutputID(ouID);


        for(int i=0;i<pre.size();i++){
            P.connect(pre[i].asInt(),post[i].asInt());
        }

        // Finalize network
        P.postConnect();
        P.randomizeWeights(weightMin,weightMax);

        // DEFINE THE TRAINING PATTERNS (INPUTS AND OUTPUTS)
        P.setMapInput(inVals);
        P.setMapOutput(ouVals);

        // TRAIN THE NETWORK
        std::vector<double> Error = P.run(T,errorSamplePeriod);

        // TEST THE NETWORK
        std::vector<double> Response;
        for(int i=0;i<P.output.size();i++){
            P.setInput(P.inputID,P.input[i]);
            P.convergeForward();
            for(int j=0;j<P.output[i].size();j++){
                std::cout<<"target="<<P.output[i][j]<<", output="<<P.X[P.outputID[j]]<<std::endl;
                Response.push_back(P.X[P.outputID[j]]);
            }
        }

        std::stringstream fname;
        fname << logpath << "/out.h5";
        morph::HdfData data(fname.str());
        std::stringstream path;
        path.str(""); path.clear(); path << "/error";
        data.add_contained_vals (path.str().c_str(), Error);
        path.str(""); path.clear(); path << "/response";
        data.add_contained_vals (path.str().c_str(), Response);

        return 0.;
}
