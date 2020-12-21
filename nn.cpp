#include <iostream>
#include <vector>
#include <cstdlib>
#include <deque>
#include <cmath>
#include <random>
using namespace std;

uniform_real_distribution<double> unif(0,1);
default_random_engine re;
double a_random_double = unif(re);


double relu(double z){
	if (z>=0.0)
		return z;
	return 0;
}

double sigmoid(double z){
	return (double(1)/(1+exp(-z)));
}

class neuron{
public:
	double z= unif(re),a,b ;
	//a -> activation 
	//b -> error in backward prop
	vector<double> owt,iwt;

	neuron(int output_size,int input_size){
		owt.resize(output_size,double(unif(re)));
		iwt.resize(input_size); // no of neurons in prev layer
	}
};

deque <neuron> nodeCache;

neuron* new_neuron(int output_size,int input_size){
	nodeCache.push_back(neuron(output_size,input_size));
	neuron* node = &nodeCache.back();
	return node;
}

class Dense{
public:
	int layer_size;
	vector <neuron*> N;
	Dense(int layer_size,int owt_size,int iwt_size){
		//layer_size = input_size for the layer
		//owt_size = size of wt_vector for neuron = layer_size of next layer
		//iwt_size = size of wt_vector for neuron = layer_size of prev layer
		this->layer_size = layer_size;
		N.resize(layer_size,NULL);
		for (int i=0;i<layer_size;++i){
			N[i]=new_neuron(owt_size,iwt_size);
		}
	}
};

void Activate(Dense* &layer){
	for (int i=0;i<layer->N.size();++i){
		layer->N[i]->a = relu(layer->N[i]->z);
	}
}


void forward(vector<Dense*> &Net,vector<double> ip){
	// feed input into first layer
	for (int i=0;i<ip.size();++i){
		Net[0]->N[i]->z = ip[i];
		Net[0]->N[i]->a = relu(ip[i]);
	}

	// propagate
	// for each layer in net
	for (int i=0;i<Net.size()-1;++i)
	{
		// for each neuron 
		for (int j=0;j<Net[i]->N.size();++j){
			// multiply owt to activation and add to activation of neuron of next layer
			for (int k=0;k<Net[i]->N[j]->owt.size();++k){
				Net[i+1]->N[k]->z += (Net[i]->N[j]->owt[k] * Net[i]->N[j]->a);
				Net[i+1]->N[k]->a = relu(Net[i+1]->N[k]->z);
				Net[i+1]->N[k]->iwt[j] = Net[i]->N[j]->owt[k];
			}
		}
	}	
}


void backprop(vector<Dense*>&Net,vector<double> op,vector<double>pred)
{
	for (int i=0;i<Net.back()->N.size();++i){
		Net.back()->N[i]->b = pred[i]-op[i];
	}

	//for each layer
	for (int i=Net.size()-1;i>0;--i){
		//for each neuron
		for (int j=0;j<Net[i]->N.size();++j){
			// multiply iwt to error and add to error of neuron of prev layer
			for (int k=0;k<Net[i]->N[j]->iwt.size();++k){
				Net[i]->N[j]->iwt[k] = (Net[i]->N[j]->iwt[k]*Net[i]->N[j]->b);				
				Net[i-1]->N[k]->owt[j] = Net[i]->N[j]->iwt[k];
				Net[i-1]->N[k]->b += (Net[i]->N[j]->iwt[k]*Net[i]->N[j]->b);
				
			}
		}
	}
}

vector<double> Pred(vector<Dense*> Net){
	vector<double> v;

	for (int i=0;i<Net.back()->N.size();++i){
		v.push_back(sigmoid(Net.back()->N[i]->a));
	}

	return v;
}

void printPred(vector<Dense*> Net){
	for (int i=0;i<Net.back()->N.size();++i){
		cout<<sigmoid(Net.back()->N[i]->a)<<" ";
	}
	cout<<"\n";
}

int main(){
	// 3 4 4 2
	vector<double> ip;ip = {1.0,-2.0,3.0};
	vector<double> op;op = {0.1,0.5,0.9};
	vector<Dense*> Net;
	Net.push_back(new Dense(3,4,0));
	Net.push_back(new Dense(4,4,3));
	Net.push_back(new Dense(4,2,4));
	Net.push_back(new Dense(2,0,4));

	for (int i=0;i<10;++i){
		cout<<"epoch = "<<i<<"\n";
		forward(Net,ip);
		backprop(Net,op,Pred(Net));
		cout<<"Prediction = ";printPred(Net);cout<<"\n";
	}
	

	return 0;
}