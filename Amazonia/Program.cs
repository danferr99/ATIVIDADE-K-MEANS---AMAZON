using System;
using System.IO;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace AmazonAnalitcs
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "amazonia.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "AmazonCluster.zip");


        static void Main(string[] args)
        {

            var mlContext = new MLContext(seed: 0);



            IDataView dataView = mlContext.Data.LoadFromTextFile<AmazonData>(_dataPath, hasHeader: false, separatorChar: '"');


            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "year", "state", "month", "number","date")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));



            var model = pipeline.Fit(dataView);



            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }



            var predictor = mlContext.Model.CreatePredictionEngine<AmazonData, ClusterPrediction>(model);



            var prediction = predictor.Predict(TestAmazonData.Janeiro);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");

        }
    }
}