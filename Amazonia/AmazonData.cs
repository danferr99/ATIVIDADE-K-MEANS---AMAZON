using Microsoft.ML.Data;

namespace AmazonAnalitcs
{
    public class AmazonData
    {
        [LoadColumn(0)]
        public int year;

        [LoadColumn(1)]
        public string state;

        [LoadColumn(2)]
        public string month;

        [LoadColumn(3)]
        public int number;

        [LoadColumn(4)]
        public int date;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}