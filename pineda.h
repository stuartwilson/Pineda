
morph::RandUniform<double, std::mt19937>* rng;

class Network{

public:

    int N, Nplus1, maxSteps, nWeight;
    std::vector<double> X, Y, U, J, F, Fprime, W, Wbest;
    std::vector<double> V, Input;
    std::vector<int> Pre, Post, inputID, outputID;
    std::vector<std::vector<double> > input, output;

    double dt, dtOverTauX, dtOverTauY, dtOverTauW, divThresh;

    Network(double dt, int N, double tauX, double tauY, double tauW, double divThresh, int maxSteps){

        this->N = N;
        this->Nplus1 = N;

        this->dt = dt;
        this->dtOverTauX = dt/tauX;
        this->dtOverTauY = dt/tauY;
        this->dtOverTauW = dt/tauW;

        this->divThresh = divThresh * N;
        this->maxSteps = maxSteps;

        X.resize(N);
        Y.resize(N);
        U.resize(N);
        J.resize(N);
        F.resize(N);
        Fprime.resize(N);
        Pre.resize(0);
        Post.resize(0);
        W.resize(0);

    }

    void setInputID(std::vector<int> x){
        this->inputID = x;
    }

    void setOutputID(std::vector<int> x){
        this->outputID = x;
    }

    void setMapInput(std::vector<std::vector<double> > x){
        this->input = x;
    }

    void setMapOutput(std::vector<std::vector<double> > x){
        this->output = x;
    }

    void addBias(void){
        for(int i=0;i<N;i++){
            W.push_back(0.0);
            Pre.push_back(N);
            Post.push_back(i);
        }
        X.push_back(1.0); // No similar operation for Y?
        Nplus1 = N+1;
        V.resize(Nplus1,0.0);
        Input.resize(Nplus1,0.0);
    }

    void connect(int pre, int post, double weight=0.0){
        W.push_back(weight);
        Pre.push_back(pre);
        Post.push_back(post);
    }

    void randomizeWeights(double weightMin, double weightMax){
        double weightRange = weightMax-weightMin;
        for(int i=0;i<W.size();i++){
            W[i] = rng->get()*weightRange+weightMin;
        }
    }

    void setNet(void){
        nWeight = W.size();
        Wbest = W;
    }

    void postConnect(void){
        addBias();
        setNet();
    }

    void reset(void){
        for(int i=0;i<X.size()-1;i++){
            X[i] = 0.0;
        }
        for(int i=0;i<Y.size()-1;i++){ // but Y not resized in same way as X above?
            Y[i] = 0.0;
        }
        for(int i=0;i<Input.size();i++){
            Input[i] = 0.0;
        }
    }

    void forward(void){
        for(int i=0;i<U.size();i++){
            U[i] = 0.0;
        }
        for(int i=0;i<nWeight;i++){
            U[Post[i]] = U[Post[i]] + X[Pre[i]]*W[i];
        }

        for(int i=0;i<N;i++){
            F[i] = 1./(1.0+exp(-U[i]));
        }

        for(int i=0;i<N;i++){
            X[i] += dtOverTauX * (-X[i]+F[i]+Input[i]);
        }
    }


    void setError(std::vector<int> oID, std::vector<double> targetOutput){
        for(int i=0;i<J.size();i++){
            J[i] = 0.0;
        }
        for(int i=0;i<oID.size();i++){
            J[oID[i]] = targetOutput[i] - X[oID[i]];
        }
    }

    void backward(void){

        for(int i=0;i<F.size();i++){
            Fprime[i] = F[i]*(1.0 - F[i]);
        }
        for(int i=0;i<V.size()-1;i++){
            V[i] = 0.0;
        }
        for(int i=0;i<nWeight;i++){
            V[Pre[i]] = V[Pre[i]] + Fprime[Post[i]]*W[i]*Y[Post[i]];
        }
        for(int i=0;i<N;i++){
            Y[i] += dtOverTauY * (V[i]-Y[i]+J[i]);
        }

    }

    void weightUpdate(void){
        for(int i=0;i<nWeight;i++){
            double delta = X[Pre[i]] * Y[Post[i]] * Fprime[Post[i]];
            if(delta<-1.0){
                W[i] -= dtOverTauW;
            } else if (delta>1.0){
                W[i] += dtOverTauW;
            } else {
                W[i] += dtOverTauW*delta;
            }
        }

    }


    double getError(void){
        double sum = 0.0;
        for (int i=0;i<J.size();i++){
            sum += J[i]*J[i];
        }
        return sum*0.5;
    }


    bool convergeForward(void){
        std::vector<double> Xpre(N,0.0);
        double total = (double)N;
        for(int t=0;t<maxSteps;t++){
            if(total>divThresh){
                Xpre=X;
                forward();
                total = 0.0;
                for(int i=0;i<N;i++){
                    total += (X[i]-Xpre[i])*(X[i]-Xpre[i]);
                }
            } else {
                return true;
            }
        }
        return false;
    }


    bool convergeBackward(void){
        std::vector<double> Ypre(N,0.0);
        double total = (double)N;
        for(int t=0;t<maxSteps;t++){
            if(total>divThresh){
                Ypre=Y;
                backward();
                total = 0.0;
                for(int i=0;i<N;i++){
                    total += (Y[i]-Ypre[i])*(Y[i]-Ypre[i]);
                }
            } else {
                return true;
            }
        }
        return false;
    }

    void setInput(std::vector<int> inputID, std::vector<double> inputVal){
        reset();
        for(int i=0;i<inputID.size();i++){
            Input[inputID[i]] = inputVal[i];
        }
    }

    std::vector<double> run(int K, int errorSamplePeriod){
        std::vector<double> Error;
        double errMin = 1e9;
        for(int k=0;k<K;k++){
            if(k%errorSamplePeriod){
                int i = floor(rng->get()*output.size());
                setInput(inputID,input[i]);
                convergeForward();
                setError(outputID,output[i]);
                convergeBackward();
                weightUpdate();
            } else {
                double err = 0.0;
                int count = 0;
                for(int i=0;i<output.size();i++){
                    setInput(inputID,input[i]);
                    convergeForward();
                    setError(outputID,output[i]);
                    err += getError();
                    count ++;
                }
                err /= (double)count;
                if(err<errMin){
                    errMin = err;
                }
                if (err>2.0*errMin){
                    W = Wbest;
                } else {
                    Wbest = W;
                }

                Error.push_back(err);

                if(k%(K/100)==0){
                    std::cout<<"progress: "<<100.0*k/K<<"%    \r"<<std::flush;
                }
            }
        }

        W = Wbest;
        return Error;
    }

};
