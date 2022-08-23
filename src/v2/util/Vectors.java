package v2.util;

public class Vectors {
    public static double scalarProduct(double[] a, double[] b) {
        if(a.length!=b.length) throw new IllegalArgumentException("Vettori di dimensione non coerente");
        double sum=0;
        for(int i=0;i<a.length;++i)
            sum+=a[i]*b[i];

        return sum;
    }
    public static double squaredError(double[] a, double[] b) {
        if(a.length!=b.length) throw new IllegalArgumentException("Vettori di dimensione non coerente");
        double sum=0;
        for(int i=0;i<a.length;++i)
            sum+=Math.pow(a[i]-b[i],2);
        return Math.sqrt(sum)/a.length;
    }

    public static double[] convertToPositional(int label) {
        double[] ret=new double[10];
        for(int i=0;i<10;++i) {
            if(i==label) ret[i]=1;
            else ret[i]=0;
        }
        return ret;
    }

    public static int getMaxIndex(double[] evaluate) {
        double max=evaluate[0];
        int ret=0;
        for(int i=0;i<evaluate.length;++i)
            if(evaluate[i]>max) {
                max=evaluate[i];
                ret=i;
            }
        return ret;
    }
}
