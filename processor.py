"""Quick Reference:
 https://github.com/ThomIves/BasicLinearAlgebraToolsPurePy/blob/master/LinearAlgebraPurePython.py"""


def new_matrix(rows, cols):
    """Create matrix of zeros of dimensions rows, cols
    :param rows: Number of rows
    :param cols: Number of cols
    :return A matrix of zeros"""
    zeroes_matrix = []
    while len(zeroes_matrix) < rows:
        zeroes_matrix.append([])
        while len(zeroes_matrix[-1]) < cols:
            zeroes_matrix[-1].append(0)
    return zeroes_matrix


def print_matrix(matrix):
    """Print matrix of form [[0,0],[0,0],[0,0]...] into
    0 0
    0 0
    0 0
    ...
    param: matrix: Matrix to print"""
    for row in matrix:
        print(*[v for v in row])


def dimensions(matrix):
    """Return dimensions of matrix in form rows, cols
    :param matrix: Matrix
    :return rows,cols"""
    return len(matrix), len(matrix[0])


def float_or_int(number):
    """Check if a string contains a float or an int and return string converted
    :param number: Number to check
    :return float(number) or int(number)"""
    if number.isdigit():
        return int(number)
    return float(number)


def input_matrix(ordinal):
    """Just for handling matrix input
    :param ordinal: format input to ordinal value e.g.(first, second)"""
    ordinals = {1: 'first', 2: 'second', 3: 'third', 0: ''}
    ordinal = ordinals[ordinal]

    rows, cols = input(f'Enter size of {ordinal} matrix:').split()
    rows = int(rows)
    cols = int(cols)
    matrix = new_matrix(rows, cols)
    print(f'Enter {ordinal} matrix:')
    # n Index of current row
    for n in range(len(matrix)):
        row = input().split()
        #  m Index of current value
        for m in range(len(matrix[0])):
            # Current number
            matrix[n][m] = float_or_int(row[m])
    return matrix


def matrix_addition(matrix1, matrix2):
    """"Handle Matrix Addition
    :param matrix1
    :param matrix2
    * Matrices need to be same dimensions"""
    # Step 1
    # Get dimensions matrix 1
    number_rows1, number_cols1 = dimensions(matrix1)
    # Get dimensions matrix 2
    number_rows2, number_cols2 = dimensions(matrix2)
    # Handle different dimensions
    if number_cols1 != number_cols2 or number_rows1 != number_rows2:
        print('The operation cannot be performed.')
        main()
    # Step 2
    res = new_matrix(number_rows1, number_cols2)
    # Perform 1 by 1 sum
    for n in range(number_rows1):
        for m in range(number_cols2):
            res[n][m] = matrix1[n][m] + matrix2[n][m]
    return res


def mul_by_n(matrix, i):
    """Multiply matrix by i
    :param matrix: Matrix to multiply
    :param i: Number to multiply matrix
    :return Matrix multiplied by n"""
    # Step 1: Obtain dimensions
    rows, cols = dimensions(matrix)
    # Step 2
    for n in range(rows):
        for m in range(cols):
            matrix[n][m] *= i

    return matrix


def multiply_matrices(matrix1, matrix2):
    """Multiply two matrices
    :param matrix1: First Matrix
    :param matrix2: Second Matrix
    :return Product of both matrices"""
    # Step 1: Obtain dimensions for both matrices
    number_rows1, number_cols1 = dimensions(matrix1)
    number_rows2, number_cols2 = dimensions(matrix2)
    # Step 2: Do multiplication and handle errors
    if number_cols1 != number_rows2:
        print('The operation cannot be performed.')
    res = new_matrix(number_rows1, number_cols2)
    for n in range(number_rows1):
        for mm in range(number_cols2):
            pres = 0
            for m in range(number_cols1):
                pres += matrix1[n][m] * matrix2[m][mm]
                res[n][mm] = pres
    return res


def transpose():
    """User menu to interact with transpose options
    :return Result of select transpose"""

    op = int(input("""1. Main diagonal
2. Side diagonal
3. Vertical line
4. Horizontal line
Your choice:"""))
    matrix = input_matrix(0)
    res = 0
    if op == 1:
        res = transpose_main(matrix)
    elif op == 2:
        res = transpose_side(matrix)
    elif op == 3:
        res = transpose_vertical(matrix)
    elif op == 4:
        res = transpose_horizontal(matrix)
    return res


def transpose_main(matrix):
    """Transpose matrix against main diagonal
    :param matrix: Matrix to transpose
    :return Transposed Matrix"""
    # Step 1 Gather dimensions
    rows, cols = dimensions(matrix)
    # Step
    # 2 transpose
    matrixt = new_matrix(cols, rows)
    for n in range(rows):
        for m in range(cols):
            matrixt[m][n] = matrix[n][m]
    return matrixt


def transpose_side(matrix):
    """Transpose matrix against side diagonal
    :param matrix: Matrix to transpose
    :return Transposed Matrix"""
    # Step 1 reverse matrix
    rows, cols = dimensions(matrix)
    for n in range(rows):
        matrix[n] = matrix[n][::-1]
    matrix = matrix[::-1]
    # Step 2 Pass reversed matrix to transpose_main
    return transpose_main(matrix)


def transpose_vertical(matrix):
    """Transpose matrix against vertical line
    :param matrix: Matrix to transpose
    :return Transposed Matrix"""
    for n in range(len(matrix)):
        matrix[n] = matrix[n][::-1]
    return matrix


def transpose_horizontal(matrix):
    """Transpose matrix against horizontal diagonal
    :param matrix: Matrix to transpose
    :return Transposed Matrix"""
    return matrix[::-1]


# not mine check reference
def calc_determinant(matrix, total=0):
    """
    Find determinant of a square matrix using full recursion
        :param matrix: the matrix to find the determinant for
        :param total: safely establish a total at each recursion level
        :returns: the running total for the levels of recursion
    """
    rows, cols = dimensions(matrix)
    # Section 1: store indices in list for flexible row referencing
    indices = list(range(len(matrix)))
    # NEW 1 by 1 matrix
    if rows == 1 and cols == 1:
        return matrix[0][0]
    # Section 2: when at 2x2 submatrices recursive calls end
    if rows == 2 and cols == 2:
        val = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return val

    # Section 3: define submatrix for focus column and call this function
    for fc in indices:  # for each focus column, find the submatrix ...
        As = matrix[:]   # make a copy, and ...
        As = As[1:]  # ... remove the first row
        height = len(As)

        for i in range(height):  # for each remaining row of submatrix ...
            As[i] = As[i][0:fc] + As[i][fc+1:]  # zero focus column elements

        sign = (-1) ** (fc % 2)  # alternate signs for submatrix multiplier
        sub_det = calc_determinant(As)  # pass submatrix recursively
        total += sign * matrix[0][fc] * sub_det  # total all returns from recursion

    return total


# More copies :)
def inverse(matrix, tol=0):
    # Section 2: Make copies of A & I, AM & IM, to use for row ops
    n = len(matrix)
    AM = matrix[:]
    I = identity_matrix(n)
    IM = I[:]

    # Section 3: Perform row operations
    indices = list(range(n))  # to allow flexible row referencing ***
    for fd in range(n):  # fd stands for focus diagonal
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse.
        for j in range(n):  # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd + 1:]:
            # *** skip row with fd in it.
            crScaler = AM[i][fd]  # cr stands for "current row".
            for j in range(n):
                # cr - crScaler * fdRow, but one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
    # Section 4: Make sure IM is an inverse of A with specified tolerance
    if check_matrix_equality(I,multiply_matrices(matrix,IM),tol):
        return IM
    else:
        print("This matrix doesn't have an inverse.")


def identity_matrix(n):
    matrix = new_matrix(n, n)
    for p in range(n):
        matrix[p][p] = 1
    return matrix


def check_matrix_equality(A, B, tol=None):
    """
    Checks the equality of two matrices.
        :param A: The first matrix
        :param B: The second matrix
        :param tol: The decimal place tolerance of the check
        :return: The boolean result of the equality check
    """
    # Section 1: First ensure matrices have same dimensions
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False

    # Section 2: Check element by element equality
    #            use tolerance if given
    for i in range(len(A)):
        for j in range(len(A[0])):
            if tol is None:
                if A[i][j] != B[i][j]:
                    return False
            else:
                if round(A[i][j], tol) != round(B[i][j], tol):
                    return False

    return True


def main():
    """User menu :)"""
    res = 0
    op = int(input("""1. Add matrices
2. Multiply matrix by a constant
3. Multiply matrices
4. Transpose Matrix
5. Calculate a determinant
6. Inverse matrix
0. Exit
Your choice:"""))

    if op == 1:
        # Addition
        ma1 = input_matrix(1)
        ma2 = input_matrix(2)
        res = matrix_addition(ma1, ma2)
    elif op == 2:
        # Multiplication by constant
        ma1 = input_matrix(1)
        n = int(input())
        res = mul_by_n(ma1, n)
    elif op == 3:
        # Multiplication of matrices
        ma1 = input_matrix(1)
        ma2 = input_matrix(2)
        res = multiply_matrices(ma1, ma2)
    elif op == 4:
        # Transpose
        res = transpose()
    elif op == 5:
        # Determinant
        ma1 = input_matrix(0)
        res = calc_determinant(ma1)
    elif op == 6:
        ma1 = input_matrix(0)
        res = inverse(ma1)
    else:
        exit()
    print('The result is:')
    if op == 5:
        print(res)
    else:
        print_matrix(res)
    main()


if __name__ == '__main__':
    main()
