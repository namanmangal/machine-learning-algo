#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Linear Regression Class
class LinearRegression {
private:
    double slope;
    double intercept;

public:
    // Training function
    void train(const std::vector<double>& x, const std::vector<double>& y) {
        double sumX = 0.0;
        double sumY = 0.0;
        double sumXY = 0.0;
        double sumX2 = 0.0;
        const int n = x.size();

        // Compute the sums
        for (int i = 0; i < n; ++i) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
        }

        // Calculate the slope and intercept
        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        intercept = (sumY - slope * sumX) / n;
    }

    // Prediction function
    double predict(double x) const {
        return slope * x + intercept;
    }

    // Calculate R-squared
    double calculateRSquared(const std::vector<double>& x, const std::vector<double>& y) const {
        const int n = x.size();
        double sumY = 0.0;
        double sumYhat = 0.0;
        double sumYDiff = 0.0;
        double sumYDiffSquared = 0.0;

        for (int i = 0; i < n; ++i) {
            double yPred = predict(x[i]);
            sumY += y[i];
            sumYhat += yPred;
            sumYDiff += y[i] - yPred;
            sumYDiffSquared += pow(y[i] - yPred, 2);
        }

        double meanY = sumY / n;
        double ssTotal = 0.0;
        double ssResidual = 0.0;

        for (int i = 0; i < n; ++i) {
            ssTotal += pow(y[i] - meanY, 2);
            ssResidual += pow(y[i] - predict(x[i]), 2);
        }

        double rSquared = 1.0 - (ssResidual / ssTotal);
        return rSquared;
    }
};

// Decision Tree Class
class DecisionTree {
private:
    double threshold;

public:
    // Training function
    void train(const std::vector<double>& x, const std::vector<double>& y) {
        double sumX = 0.0;
        double sumY = 0.0;
        const int n = x.size();

        // Compute the sums
        for (int i = 0; i < n; ++i) {
            sumX += x[i];
            sumY += y[i];
        }

        // Calculate the threshold as the average of input feature values
        threshold = sumX / n;
    }

    // Prediction function
    double predict(double x) const {
        return (x <= threshold) ? 0.0 : 1.0;
    }

    // Calculate R-squared
    double calculateRSquared(const std::vector<double>& x, const std::vector<double>& y) const {
        const int n = x.size();
        double sumY = 0.0;
        double sumYhat = 0.0;
        double sumYDiff = 0.0;
        double sumYDiffSquared = 0.0;

        for (int i = 0; i < n; ++i) {
            double yPred = predict(x[i]);
            sumY += y[i];
            sumYhat += yPred;
            sumYDiff += y[i] - yPred;
            sumYDiffSquared += pow(y[i] - yPred, 2);
        }

        double meanY = sumY / n;
        double ssTotal = 0.0;
        double ssResidual = 0.0;

        for (int i = 0; i < n; ++i) {
            ssTotal += pow(y[i] - meanY, 2);
            ssResidual += pow(y[i] - predict(x[i]), 2);
        }

        double rSquared = 1.0 - (ssResidual / ssTotal);
        return rSquared;
    }
};

int main() {
    // Create a linear regression object
    LinearRegression lr;

    // Create a decision tree object
    DecisionTree dt;

    // Get the input values from the user
    std::cout << "Enter the number of data points: ";
    int numPoints;
    std::cin >> numPoints;

    std::vector<double> x(numPoints);
    std::vector<double> y(numPoints);

    std::cout << "Enter the input feature values:\n";
    for (int i = 0; i < numPoints; ++i) {
        std::cout << "Input " << i + 1 << ": ";
        std::cin >> x[i];
    }

    std::cout << "Enter the target values:\n";
    for (int i = 0; i < numPoints; ++i) {
        std::cout << "Target " << i + 1 << ": ";
        std::cin >> y[i];
    }

    // Train the linear regression model
    lr.train(x, y);

    // Train the decision tree model
    dt.train(x, y);

    // Predict a value using linear regression
    double input;
    std::cout << "Enter a value to predict using linear regression: ";
    std::cin >> input;

    double lrPrediction = lr.predict(input);
    std::cout << "Linear Regression Prediction for input " << input << ": " << lrPrediction << std::endl;

    // Predict a value using decision tree
    std::cout << "Enter a value to predict using decision tree: ";
    std::cin >> input;

    double dtPrediction = dt.predict(input);
    std::cout << "Decision Tree Prediction for input " << input << ": " << dtPrediction << std::endl;

    // Calculate mean squared error, root mean squared error, and R-squared for linear regression
    double lrMSE = 0.0;
    double lrRMSE = 0.0;
    for (int i = 0; i < numPoints; ++i) {
        double yPred = lr.predict(x[i]);
        lrMSE += pow(yPred - y[i], 2);
    }
    lrMSE /= numPoints;
    lrRMSE = sqrt(lrMSE);
    double lrRSquared = lr.calculateRSquared(x, y);

    // Calculate mean squared error, root mean squared error, and R-squared for decision tree
    double dtMSE = 0.0;
    double dtRMSE = 0.0;
    for (int i = 0; i < numPoints; ++i) {
        double yPred = dt.predict(x[i]);
        dtMSE += pow(yPred - y[i], 2);
    }
    dtMSE /= numPoints;
    dtRMSE = sqrt(dtMSE);
    double dtRSquared = dt.calculateRSquared(x, y);

    // Output the evaluation metrics
    std::cout << "Linear Regression Evaluation:\n";
    std::cout << "Mean Squared Error (MSE): " << lrMSE << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << lrRMSE << std::endl;
    std::cout << "R-squared: " << lrRSquared << std::endl;

    std::cout << "Decision Tree Evaluation:\n";
    std::cout << "Mean Squared Error (MSE): " << dtMSE << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << dtRMSE << std::endl;
    std::cout << "R-squared: " << dtRSquared << std::endl;

    return 0;
}
