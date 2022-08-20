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
            sum+=Math.abs(a[i]-b[i]);
        return sum;
    }
}
