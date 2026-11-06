import unittest
import numpy as np


class VecFunctions:

    # Sub vector sclicing
    def subVector(A,i,j):
        print('A : ',A);
        print('i : ',i);
        print('j : ',j);
        print('a[i,j] :',A[i-1:j]);
        return A[i-1:j];

    # Scaler mean : m     
    def scalerMean(A):
        return np.sum(A) / len(A)

    # Scaler variance : v     
    def scalarVariance(A):
        m = VecFunctions.scalerMean(A)
        return sum((A - m)**2) / len(A)

class TestQues1(unittest.TestCase):

    # Normal vector
    def test_q1a(self):
        print('------------- Q1 - a (Start)-------------');
        
        x = np.array([1, 4, 9, 16, 25, 36], dtype=np.float64);
        print('x :\n',x);
        
        D = np.zeros((6, 5), dtype=np.float64);
        for j in range(5):
            D[j,j]=-1;
            D[j+1,j]=1;

        print('D :\n',D);
        print('x^T.D :\n',x.dot(D));

        output = x.dot(D)
        expected = np.array([3, 5, 7, 9, 11], dtype=np.float64)
        np.testing.assert_allclose(output, expected)
        
        print('------------- Q1 - a (End)-------------');

    def test_q1b(self):
        print('------------- Q1 - b (Start)  -------------');

        x = np.array([1, 4, 9, 16, 25, 36], dtype=np.float64);

        output = VecFunctions.subVector(x,2,5);
        expected = np.array([4, 9, 16, 25], dtype=np.float64)

        print('output :',output)
        print('expected :',expected)
        
        np.testing.assert_array_equal(output, expected)

        print('------------- Q1 - b (End)  -------------');

    def test_q1c(self):
            print('------------- Q1 - c (Start)-------------');

            x = np.array([1, 4, 9, 16, 25, 36], dtype=np.float64);
            y = VecFunctions.subVector(x,2,6);
            z = VecFunctions.subVector(x,1,5);
            D = np.zeros((6, 5), dtype=np.float64);
            for j in range(5):
                D[j,j]=-1;
                D[j+1,j]=1;

            diff=y-z;
            dot=x.dot(D)

            print('x[2:6] - x[1:5]: ',diff);
            print('x^T.D: ',x.dot(D));
            print('x^T.D:\n',type(x.dot(D)));
            np.testing.assert_almost_equal(diff, dot)
            print('------------- Q1 - c (End) -------------');
    

class TestQues2(unittest.TestCase):

    # Compute D^T.x by matrix multiplications
    def test_q2a(self):
        print('------------- Q2 - a (Start) -------------');

        x = np.random.default_rng(0).standard_normal((6, 768))
        D = np.zeros((6, 5), dtype=np.float64);
        for j in range(5):
            D[j,j]=-1;
            D[j+1,j]=1;

        print('x:\n',x)
        print('x.shape',x.shape)
        print('D:\n',D)
        print('D.shape',D.shape)
        print('D.T:\n',D.T)
        
        result = D.T.dot(x);

        print('result:\n',result)
        print('result.shape:\n',result.shape)
        self.assertEqual(x.shape, (6, 768))
        self.assertEqual(D.shape, (6, 5))
        self.assertEqual(result.shape, (5, 768))

        print('------------- Q2 - a (End) -------------');

class TestQues3(unittest.TestCase):

    # Test layer normalization given formula 
    def test_q3a(self):
        print('------------- Q3 - a (Start) -------------');
        
        e = np.random.default_rng(0).standard_normal(768).astype(np.float64);
        beta = np.zeros(768, dtype=np.float64)
        gamma = np.ones(768, dtype=np.float64)
        epsilon = 1e-5;
        m = VecFunctions.scalerMean(e)
        v = VecFunctions.scalarVariance(e)
      
        e_totem = ((e - m) / np.sqrt(v + epsilon)) * gamma + beta

        test_m = np.mean(e)
        test_v = np.var(e)
        expected = ((e - test_m) / np.sqrt(test_v + epsilon)) * gamma + beta
        np.testing.assert_allclose(e_totem, expected, 1e-7,1e-7)

        print('------------- Q3 - a (End) -------------');

    # Test and compare normalized output mean and variance from np's.
    def test_q3b(self):
        print('------------- Q3 - b (Start) -------------');

        e = np.random.default_rng(0).standard_normal(768).astype(np.float64)
        gamma = np.ones(768, dtype=np.float64)
        beta = np.zeros(768, dtype=np.float64)
        epsilon = 1e-5

        m = VecFunctions.scalerMean(e)
        v = VecFunctions.scalarVariance(e)
        e_totem = ((e - m) / np.sqrt(v + epsilon)) * gamma + beta

        test_v = np.var(e)
        expected_variance = test_v / (test_v + epsilon)

        self.assertAlmostEqual(float(np.mean(e_totem)), 0.0, places=10)
        self.assertAlmostEqual(float(np.var(e_totem)), expected_variance, places=10)

        print('------------- Q3 - b (End) -------------');

if __name__ == "__main__":
    unittest.main()