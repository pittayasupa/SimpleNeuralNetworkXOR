using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkXOR
{
    public class NeuralNetwork
    {
        public enum LayerType
        {
            None,
            Input,
            Hidden,
            Output,
        }
        int[] layer;
        Layer[] layers;
        //Weight[] weights;
        //Weight[] weightsDelta;

        public NeuralNetwork(int[] _numOfLayer)
        {
            layer = new int[_numOfLayer.Length];
            for(int i = 0; i < layer.Length; i++)
            {
                layer[i] = _numOfLayer[i];
            }
            layers = new Layer[_numOfLayer.Length];

            for(int i = 0; i < layers.Length - 1; i++)
            {
                layers[i] = new Layer(_numOfLayer[i], (i == 0 ? LayerType.Input : LayerType.Hidden), i);
            } 
            layers[layers.Length - 1] = new Layer(_numOfLayer[_numOfLayer.Length - 1], LayerType.Output, layers.Length - 1);

            for (int i = 1; i < layers.Length; i++)
            {
                if(i - 1 >= 0)
                {
                    layers[i].InitializeWeights(_numOfLayer[i - 1], _numOfLayer[i]);
                }
            } 
        }

        public void FeedForward(float[] _inputs)
        {
            //Console.WriteLine($"feed layer {0}");
            //layers[0].Forward(_inputs);
            layers[0].activates = _inputs;
            for(int i = 1; i < layers.Length; i++)
            {
                if(i - 1 >= 0)
                {
                    //Console.WriteLine($"feed layer {i}");
                    //layers[i].Forward(layers[i-1].activates, weights[i-1].weights, weights[i - 1].deltas);
                    layers[i].Forward(layers[i-1].activates);
                    //weights[i - 1].weights = layers[i].weights;
                }
            } 
        }
        public void BackProp(float[] _targets)
        {
            //layers[layers.Length - 1].BackPropOutput(_targets);  
            for(int i = layers.Length - 1; i > 0; i--)
            {
                if(i == layers.Length - 1)
                {
                    //Console.WriteLine($"BackProp Layer {i-1} <= {i}");
                    layers[i].BackPropOutput(_targets);
                }
                else
                {
                    //Console.WriteLine($"BackProp Layer {i-1} <= {i}");
                    layers[i].BackPropHidden(layers[i+1].gamma, layers[i+1].weights);
                }
            } 
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].UpdateWeights();
            }
        }
        public float[] GetOutput()
        {
            return layers[layers.Length - 1].GetOutput();
        } 

        class Layer
        {
            LayerType layerType = LayerType.None;
            int numOfInput = 0;
            int numOfOutput = 0;
            int numOfLayer = 0;

            public float[] inputs;
            public float[] outputs;
            public float[] activates;
            public float[] errors;
            public float[] gamma;

            public float[,] weights;
            public float[,] deltas;

            public static Random ran = new Random();
            public Layer(int _numOfOutput, LayerType _layerType = LayerType.Hidden, int _numOfLayer = 0)
            { 
                numOfOutput = _numOfOutput;
                layerType = _layerType;
                numOfLayer = _numOfLayer;
                Console.WriteLine($"new layer {numOfLayer} {layerType.ToString()} => {numOfOutput}");

                outputs = new float[numOfOutput];
                activates = new float[numOfOutput];
                errors = new float[numOfOutput];
                gamma = new float[numOfOutput]; 
            } 
            public void InitializeWeights(int _numOfInput, int _numOfOutput)
            {
                //Console.WriteLine($"InitializeWeights[{_numOfInput},{_numOfOutput}]");
                weights = new float[_numOfInput, _numOfOutput];
                deltas = new float[_numOfInput, _numOfOutput];

                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        weights[i, j] = (float)ran.NextDouble() - 0.0005f;
                        //Console.WriteLine($"weights[{i},{j}] = {weights[i, j]}");
                    }
                }
            }
            
            public void Forward(float[] _inputs)
            {
                numOfInput = _inputs.Length;
                inputs = _inputs;
                for (int i = 0; i < numOfOutput; i++)
                {
                    outputs[i] = 0;
                    for (int j = 0; j < numOfInput; j++)
                    {
                        //Console.WriteLine($"outputs[{i}] = inputs[{j}] * weights[{j}, {i}]");
                        //Console.WriteLine($"outputs[{i}] = {inputs[j]} * {weights[j, i]}");
                        outputs[i] += inputs[j] * weights[j, i];
                    }
                    activates[i] = Sigmoid(outputs[i]);
                    //Console.WriteLine($"outputs[{i}] = {outputs[i]}");
                    //Console.WriteLine($"activates[{i}] = {activates[i]}");
                } 
            }
            public void BackPropOutput(float[] _target)
            {
                for (int i = 0; i < numOfOutput; i++)
                {
                    errors[i] = -(_target[i] - activates[i]);
                    //Console.WriteLine($"errors[{i}] = -({_target[i]} - {activates[i]})");
                    //Console.WriteLine($"errors[{i}] = {errors[i]}");
                }
                for (int i = 0; i < numOfOutput; i++)
                {
                    gamma[i] = errors[i] * SigmoidPrime(activates[i]);
                }
                for (int i = 0; i < numOfOutput; i++)
                {
                    for (int j = 0; j < numOfInput; j++)
                    {
                        //Console.WriteLine($"deltas[{i},{j}] = gamma[{i}] * inputs[{j}]");
                        deltas[j, i] = gamma[i] * inputs[j];
                        //Console.WriteLine($"deltas[{i},{j}] = {deltas[j, i]}");
                    }
                }
            }
            public void BackPropHidden(float[] gammaForward, float[,] weightsForward)
            { 
                for (int i = 0; i < numOfOutput; i++)
                {
                    gamma[i] = 0;
                    for (int j = 0; j < gammaForward.Length; j++)
                    {
                        gamma[i] += gammaForward[j] * weightsForward[i, j];
                    }
                    gamma[i] *= SigmoidPrime(activates[i]);
                }
                for (int i = 0; i < numOfOutput; i++)
                {
                    for (int j = 0; j < numOfInput; j++)
                    {
                        deltas[j, i] = gamma[i] * inputs[j];
                    }
                } 
            }
            public void UpdateWeights()
            {
                for (int i = 0; i < numOfOutput; i++)
                {
                    for (int j = 0; j < numOfInput; j++)
                    {
                        weights[j, i] -= deltas[j, i] * 0.05f;
                    }
                }
            }
            public float[] GetOutput()
            {
                for(int i = 0; i < numOfOutput; i++)
                {
                    activates[i] = Sigmoid(outputs[i]);
                }
                return activates;
            } 
            public float Sigmoid(float x)
            {
                float exp = (float)Math.Exp(-x);
                return 1 / (1 + exp);
            }
            public float SigmoidPrime(float x)
            {
                return x * (1 - x);
            }
            public float Tanh(float x)
            {
               return (float)Math.Tanh(x);
            }
            public float TanhPrime(float x)
            {
                return 1 - (x * x);
            }
        }
    }
}
