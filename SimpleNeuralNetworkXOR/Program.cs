﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetworkXOR
{
    class Program
    {
        static void Main(string[] args)
        {

            NeuralNetwork net = new NeuralNetwork(new int[] { 3, 10, 10, 1 });

            float error = 0;
            int cost = 0;
            int epoch = 0;

            for (int e = 0; e < 100000; e++)
            {
                error = 0;
                for (int i = 0; i < 100; i++)
                {
                    net.FeedForward(new float[] { 0, 0, 0 });
                    net.BackProp(new float[] { 0 });
                    net.FeedForward(new float[] { 0, 0, 1 });
                    net.BackProp(new float[] { 1 });
                    net.FeedForward(new float[] { 0, 1, 0 });
                    net.BackProp(new float[] { 1 });
                    net.FeedForward(new float[] { 0, 1, 1 });
                    net.BackProp(new float[] { 0 });
                    net.FeedForward(new float[] { 1, 0, 0 });
                    net.BackProp(new float[] { 1 });
                    net.FeedForward(new float[] { 1, 0, 1 });
                    net.BackProp(new float[] { 0 });
                    net.FeedForward(new float[] { 1, 1, 0 });
                    net.BackProp(new float[] { 0 });
                    net.FeedForward(new float[] { 1, 1, 1 });
                    net.BackProp(new float[] { 0 });
                    cost += 8;
                }

                net.FeedForward(new float[] { 0, 0, 0 });
                error += -(0 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 0, 0, 1 });
                error += (1 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 0, 1, 0 });
                error += (1 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 0, 1, 1 });
                error += -(0 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 1, 0, 0 });
                error += (1 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 1, 0, 1 });
                error += -(0 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 1, 1, 0 });
                error += -(0 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 1, 1, 1 });
                error += -(0 - net.GetOutput()[0]);
                  
                if ((error / 8) < 0.01)
                {
                    break;
                }
                epoch++;
            }

            Console.WriteLine($"epoch => {epoch}");
            Console.WriteLine($"cost => {cost}");
            Console.WriteLine($"error => {error}");
            Console.WriteLine($"error avg => {error / 8}");

            net.FeedForward(new float[] { 0, 0, 0 });
            Console.WriteLine($"0 0 0 => 0 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 0, 0, 1 });
            Console.WriteLine($"0 0 1 => 1 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 0, 1, 0 });
            Console.WriteLine($"0 1 0 => 1 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 0, 1, 1 });
            Console.WriteLine($"0 1 1 => 0 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 1, 0, 0 });
            Console.WriteLine($"1 0 0 => 1 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 1, 0, 1 });
            Console.WriteLine($"1 0 1 => 0 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 1, 1, 0 });
            Console.WriteLine($"1 1 0 => 0 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 1, 1, 1 });
            Console.WriteLine($"1 1 1 => 0 : actual => {net.GetOutput()[0]}");

            /*
            0 0 => 0
            0 1 => 1
            1 0 => 1
             1 1 => 0
           */
            /*
            NeuralNetwork net = new NeuralNetwork(new int[] { 2, 5, 5, 1 });
           
            float error = 0;
            int cost = 0;
            int epoch = 0;

            for(int e = 0; e < 100000; e++)
            {
                error = 0;
                for (int i = 0; i < 100; i++)
                {
                    net.FeedForward(new float[] { 0, 0 });
                    net.BackProp(new float[] { 0 });
                    net.FeedForward(new float[] { 0, 1 });
                    net.BackProp(new float[] { 1 });
                    net.FeedForward(new float[] { 1, 0 });
                    net.BackProp(new float[] { 1 });
                    net.FeedForward(new float[] { 1, 1 });
                    net.BackProp(new float[] { 0 });
                    cost += 4;
                }

                net.FeedForward(new float[] { 0, 0 });
                error += -(0 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 0, 1 });
                error += (1 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 1, 0 });
                error += (1 - net.GetOutput()[0]);
                net.FeedForward(new float[] { 1, 1 });
                error += -(0 - net.GetOutput()[0]);

                if ((error / 4) < 0.01) 
                { 
                    break; 
                } 
                epoch++;
            }

            Console.WriteLine($"epoch => {epoch}");
            Console.WriteLine($"cost => {cost}");
            Console.WriteLine($"error => {error}");
            Console.WriteLine($"error avg => {error / 4}");
            
            net.FeedForward(new float[] { 0, 0 });
            Console.WriteLine($"0 0 => 0 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 0, 1 });
            Console.WriteLine($"0 1 => 1 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 1, 0 });
            Console.WriteLine($"1 0 => 1 : actual => {net.GetOutput()[0]}");
            net.FeedForward(new float[] { 1, 1 });
            Console.WriteLine($"1 1 => 0 : actual => {net.GetOutput()[0]}");
            */

            Console.ReadKey();
        }
    }
}
