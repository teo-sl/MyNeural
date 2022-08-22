package v2.activationFunction;

import java.io.Serializable;

public interface ActivationFunction extends Serializable {
    double evaluate(double x);
    double derive(double x);
}
