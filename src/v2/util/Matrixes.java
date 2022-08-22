package v2.util;

import java.util.Arrays;

public class Matrixes {
    public static String printMatrix(double[][] m)  {
        StringBuilder str = new StringBuilder(100);
        for(int i=0;i< m.length;++i) {
            for(int j=0;j<m[0].length;++j)
                str.append(m[i][j]+" ");
            str.append("\n");
        }
        return str.toString();
    }
    public static void reset(double[][] m) {
        for(int i=0;i<m.length;++i)
            Arrays.fill(m[i],0);
    }

}
