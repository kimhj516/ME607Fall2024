#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseSolver.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_Version.hpp"

void first_example()
{
    std::cout << Teuchos::Teuchos_Version() << std::endl << std::endl;
    // Create an empty matrix with no dimension
    Teuchos::SerialDenseMatrix<int,double> Empty_Matrix;
    // Create an empty 3x4 matrix
    Teuchos::SerialDenseMatrix<int,double> My_Matrix( 3, 4 );
    // Create a double-precision vector:
    Teuchos::SerialDenseVector<int,double> x(3);
    Teuchos::SerialDenseVector<int,double> y(3);
    // The matrix dimensions and strided storage information can be obtained:
    int rows = My_Matrix.numRows();  // number of rows
    int cols = My_Matrix.numCols();  // number of columns
    int stride = My_Matrix.stride(); // storage stride
    TEUCHOS_ASSERT_EQUALITY(rows, 3);
    TEUCHOS_ASSERT_EQUALITY(cols, 4);
    TEUCHOS_ASSERT_EQUALITY(stride, 3);
    // Matrices can change dimension:
    Empty_Matrix.shape( 3, 3 );      // size non-dimensional matrices
    My_Matrix.reshape( 3, 3 );       // resize matrices and save values
    // Filling matrices with numbers can be done in several ways:
    My_Matrix.random();             // random numbers
    std::cout << printMat(My_Matrix) << std::endl;
    My_Matrix.putScalar( 1.0 );      // every entry is 1.0
    std::cout << printMat(My_Matrix) << std::endl;
    My_Matrix(1,1) = 10.0;           // individual element access
    std::cout << printMat(My_Matrix) << std::endl;
    Empty_Matrix = My_Matrix;       // copy My_Matrix to Empty_Matrix
    x = 1.0;                        // every entry of vector is 1.0
    y = 1.0;
    // Basic matrix arithmetic can be performed:
    double d;
    Empty_Matrix += My_Matrix;         // Matrix addition
    Empty_Matrix.scale( 0.5 );         // Matrix scaling
    d = x.dot( y );                // Vector dot product
    std::cout << "d: " << d << std::endl;
    // The pointer to the array of matrix values can be obtained:
    double *My_Array=0, *My_Column=0;
    My_Array = My_Matrix.values();   // pointer to matrix values
    My_Column = My_Matrix[2];        // pointer to third column values
    // The norm of a matrix can be computed:
    double norm_one = My_Matrix.normOne();        // one norm
    double norm_inf = My_Matrix.normInf();        // infinity norm
    double norm_fro = My_Matrix.normFrobenius();  // frobenius norm
    std::cout << "one-norm: " << norm_one << " infinity-norm: " << norm_inf << " frobenius norm: " << norm_fro << std::endl;
    // Matrices can be compared:
    // Check if the matrices are equal in dimension and values
    if (Empty_Matrix == My_Matrix)
    {
        std::cout<< "The matrices are the same!" <<std::endl;
    }
    // Check if the matrices are different in dimension or values
    if (Empty_Matrix != My_Matrix)
    {
        std::cout<< "The matrices are different!" <<std::endl;
    }
    // A matrix can be factored and solved using Teuchos::SerialDenseSolver.
    Teuchos::SerialDenseSolver<int,double> My_Solver;
    Teuchos::SerialDenseMatrix<int,double> X(3,1), B(3,1);
    X.putScalar(1.0);
    B.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, My_Matrix, X, 0.0 );
    X.putScalar(0.0);  // Make sure the computed answer is correct.
    int info = 0;
    My_Solver.setMatrix( Teuchos::rcp( &My_Matrix, false ) );
    My_Solver.setVectors( Teuchos::rcp( &X, false ), Teuchos::rcp( &B, false ) );
    info = My_Solver.factor();
    if (info != 0)
        std::cout << "Teuchos::SerialDenseSolver::factor() returned : " << info << std::endl;
    info = My_Solver.solve();
    if (info != 0)
        std::cout << "Teuchos::SerialDenseSolver::solve() returned : " << info << std::endl;
    // A matrix can be sent to the output stream:
    std::cout<< std::endl << printMat(My_Matrix) << " " << printMat(X) << " " << printMat(B) << std::endl;
}

void second_example()
{
    const int number_of_rows = 4;
    Teuchos::SerialDenseMatrix<int, double> matrix(number_of_rows, number_of_rows);
    Teuchos::SerialDenseVector<int, double> vector(number_of_rows);
    matrix.random();
    vector.random();
    std::cout << printMat(matrix) << " " << printMat(vector) << std::endl;

    // Create an instance of the LAPACK class for double-precision routines
    Teuchos::LAPACK<int, double> lapack;

    // Perform a LU factorization of this matrix
    int ipiv[number_of_rows];
    int info = 0;
    char TRANS = 'N';
    lapack.GETRF(number_of_rows, number_of_rows, matrix.values(), matrix.stride(), ipiv, &info);
    lapack.GETRS(TRANS, number_of_rows, 1, matrix.values(), matrix.stride(), ipiv, vector.values(), vector.stride(), &info);
    std::cout << "matrix information: " << matrix.numRows() << " " << matrix.numCols() << " " << matrix.stride() << " " << printMat(matrix) << std::endl;
    std::cout << "vector information: " << vector.numRows() << " " << vector.numCols() << " " << vector.stride() << " " << printMat(vector) << std::endl;

    // Perform a QR factorization of this matrix
    int lwork = number_of_rows * number_of_rows;
    Teuchos::SerialDenseVector<int, double> matrix_q(number_of_rows);
    Teuchos::SerialDenseVector<int, double> matrix_r(lwork);
    lapack.GEQRF(number_of_rows, number_of_rows, matrix.values(), matrix.stride(), matrix_q.values(), matrix_r.values(), lwork, &info);
    std::cout << "matrix Q information: " << matrix_q.numRows() << " " << matrix_q.numCols() << " " << matrix_q.stride() << " " << printMat(matrix_q) << std::endl;
    std::cout << "matrix R information: " << matrix_r.numRows() << " " << matrix_r.numCols() << " " << matrix_r.stride() << " " << printMat(matrix_r) << std::endl;
    lapack.GEQR2(number_of_rows, number_of_rows, matrix.values(), matrix.stride(), matrix_q.values(), matrix_r.values(), &info);
    std::cout << "matrix Q information: " << matrix_q.numRows() << " " << matrix_q.numCols() << " " << matrix_q.stride() << " " << printMat(matrix_q) << std::endl;
    std::cout << "matrix R information: " << matrix_r.numRows() << " " << matrix_r.numCols() << " " << matrix_r.stride() << " " << printMat(matrix_r) << std::endl;
}


// main routine to call various routines
int main (int argc, char *argv[])
{
    std::cout << "=======================================" << std::endl;
    std::cout << "=========  first example  =============" << std::endl;
    first_example();
    std::cout << "=======================================" << std::endl;
    std::cout << "=========  second example  =============" << std::endl;
    second_example();
    return 0;
}
