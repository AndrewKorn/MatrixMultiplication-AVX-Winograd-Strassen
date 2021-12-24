#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include <ctime>

int N = 4096;

class Matrix {
public:
    float* data;
    float* dataT;
    int	dim;
    int	oRows;
    int	oCols;

    Matrix(float* dat, float* datT, const int n, const int offsetM, const int offsetN) : data(dat), dataT(datT), dim(n), oRows(offsetM), oCols(offsetN) {}

    inline
    float& operator()(const int row, const int col) const {
        return data[(row+oRows) * N + col + oCols];
    }

    inline
    void set(const int row, const int col, const __m256& v) const{
        _mm256_store_ps(&data[(row + oRows) * N + col + oCols], v);
    }

    inline
    void  setT(const int row, const int col, const __m256& v) const{
        _mm256_store_ps(&dataT[(col + oCols) * N + row + oRows], v);
    }

    inline
    __m256* get(const int row, const int col) const {
        return (__m256*)(&data[(row + oRows) * N + col + oCols]);
    }

    inline
    __m256* getT(const int row, const int col) const {
        return (__m256*)(&dataT[(col + oCols) * N + row + oRows]);
    }

    inline
    Matrix getSubMatrix(const int row, const int col, const int dimC) const {
        return Matrix(data, dataT, dimC, row+oRows, col + oCols);
    }

    inline
    int getDim() const {
        return dim;
    }
};

static const int	ALIGNMENT = 0x40;
static const int	TRUNCATION_POINT = 600;

inline
void Transpose8x8Shuff(Matrix& A) {
    __m256 *inI = reinterpret_cast<__m256 *>(A.data);
    __m256 rI[8];
    int offset = N / 8;
    int oRow = A.oRows;
    int oCol = A.oCols / 8;
    rI[0] = _mm256_unpacklo_ps(inI[(0 + oRow) * offset + oCol], inI[(1 + oRow) * offset + oCol]);
    rI[1] = _mm256_unpackhi_ps(inI[(0 + oRow) * offset + oCol], inI[(1 + oRow) * offset + oCol]);
    rI[2] = _mm256_unpacklo_ps(inI[(2 + oRow) * offset + oCol], inI[(3 + oRow) * offset + oCol]);
    rI[3] = _mm256_unpackhi_ps(inI[(2 + oRow) * offset + oCol], inI[(3 + oRow) * offset + oCol]);
    rI[4] = _mm256_unpacklo_ps(inI[(4 + oRow) * offset + oCol], inI[(5 + oRow) * offset + oCol]);
    rI[5] = _mm256_unpackhi_ps(inI[(4 + oRow) * offset + oCol], inI[(5 + oRow) * offset + oCol]);
    rI[6] = _mm256_unpacklo_ps(inI[(6 + oRow) * offset + oCol], inI[(7 + oRow) * offset + oCol]);
    rI[7] = _mm256_unpackhi_ps(inI[(6 + oRow) * offset + oCol], inI[(7 + oRow) * offset + oCol]);

    __m256 rrF[8];
    __m256 *rF = reinterpret_cast<__m256 *>(rI);
    rrF[0] = _mm256_shuffle_ps(rF[0], rF[2], _MM_SHUFFLE(1,0,1,0));
    rrF[1] = _mm256_shuffle_ps(rF[0], rF[2], _MM_SHUFFLE(3,2,3,2));
    rrF[2] = _mm256_shuffle_ps(rF[1], rF[3], _MM_SHUFFLE(1,0,1,0));
    rrF[3] = _mm256_shuffle_ps(rF[1], rF[3], _MM_SHUFFLE(3,2,3,2));
    rrF[4] = _mm256_shuffle_ps(rF[4], rF[6], _MM_SHUFFLE(1,0,1,0));
    rrF[5] = _mm256_shuffle_ps(rF[4], rF[6], _MM_SHUFFLE(3,2,3,2));
    rrF[6] = _mm256_shuffle_ps(rF[5], rF[7], _MM_SHUFFLE(1,0,1,0));
    rrF[7] = _mm256_shuffle_ps(rF[5], rF[7], _MM_SHUFFLE(3,2,3,2));

    rF = reinterpret_cast<__m256 *>(A.data);
    rF[(0 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[0], rrF[4], 0x20);
    rF[(1 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[1], rrF[5], 0x20);
    rF[(2 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[2], rrF[6], 0x20);
    rF[(3 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[3], rrF[7], 0x20);
    rF[(4 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[0], rrF[4], 0x31);
    rF[(5 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[1], rrF[5], 0x31);
    rF[(6 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[2], rrF[6], 0x31);
    rF[(7 + oRow) * offset + oCol] = _mm256_permute2f128_ps(rrF[3], rrF[7], 0x31);
}

inline
void TR(Matrix& A) {
    int dim = A.getDim();
    if (dim == 8) {
        Transpose8x8Shuff(A);
    }
    else {
        int dim2 = dim / 2;
        Matrix A1 = A.getSubMatrix(0, 0, dim2);
        Matrix A2 = A.getSubMatrix(0, dim2, dim2);
        Matrix A3 = A.getSubMatrix(dim2, 0, dim2);
        Matrix A4 = A.getSubMatrix(dim2, dim2, dim2);
        TR(A1);
        TR(A2);
        TR(A3);
        TR(A4);
        __m256* a = A.get(0, 0);
        for (int i = 0; i < dim2; ++i) {
            for (int j = 0; j < dim2 / 8; ++j) {
                __m256 tmp = a[i * N / 8 + j + dim2 / 8];
                a[i * N / 8 + j + dim2 / 8] = a[(i + dim2) * N / 8 + j];
                a[(i + dim2) * N / 8 + j] = tmp;
            }
        }
    }
}

inline
void naive(const Matrix& A, const Matrix& B, Matrix& C){
    int dim = A.getDim();

    for (int n = 0; n < dim; n += 8) {
        for (int m = 0; m < dim; m += 8) {
            __m256*	pA0 = A.get(n, 0);
            __m256*	pA1 = A.get(n+1, 0);
            __m256*	pA2 = A.get(n+2, 0);
            __m256*	pA3 = A.get(n+3, 0);
            __m256*	pA4 = A.get(n + 4, 0);
            __m256*	pA5 = A.get(n + 5, 0);
            __m256*	pA6 = A.get(n + 6, 0);
            __m256*	pA7 = A.get(n + 7, 0);

            __m256*	pB0 = B.getT(0, m);
            __m256*	pB1 = B.getT(0, m + 1);
            __m256*	pB2 = B.getT(0, m + 2);
            __m256*	pB3 = B.getT(0, m + 3);
            __m256*	pB4 = B.getT(0, m + 4);
            __m256*	pB5 = B.getT(0, m + 5);
            __m256*	pB6 = B.getT(0, m + 6);
            __m256*	pB7 = B.getT(0, m + 7);

            __m256	C00 = _mm256_setzero_ps();
            __m256	C01 = _mm256_setzero_ps();
            __m256	C02 = _mm256_setzero_ps();
            __m256	C03 = _mm256_setzero_ps();
            __m256	C04 = _mm256_setzero_ps();
            __m256	C05 = _mm256_setzero_ps();
            __m256	C06 = _mm256_setzero_ps();
            __m256	C07 = _mm256_setzero_ps();

            __m256	C10 = _mm256_setzero_ps();
            __m256	C11 = _mm256_setzero_ps();
            __m256	C12 = _mm256_setzero_ps();
            __m256	C13 = _mm256_setzero_ps();
            __m256	C14 = _mm256_setzero_ps();
            __m256	C15 = _mm256_setzero_ps();
            __m256	C16 = _mm256_setzero_ps();
            __m256	C17 = _mm256_setzero_ps();

            __m256	C20 = _mm256_setzero_ps();
            __m256	C21 = _mm256_setzero_ps();
            __m256	C22 = _mm256_setzero_ps();
            __m256	C23 = _mm256_setzero_ps();
            __m256	C24 = _mm256_setzero_ps();
            __m256	C25 = _mm256_setzero_ps();
            __m256	C26 = _mm256_setzero_ps();
            __m256	C27 = _mm256_setzero_ps();

            __m256	C30 = _mm256_setzero_ps();
            __m256	C31 = _mm256_setzero_ps();
            __m256	C32 = _mm256_setzero_ps();
            __m256	C33 = _mm256_setzero_ps();
            __m256	C34 = _mm256_setzero_ps();
            __m256	C35 = _mm256_setzero_ps();
            __m256	C36 = _mm256_setzero_ps();
            __m256	C37 = _mm256_setzero_ps();

            __m256	C40 = _mm256_setzero_ps();
            __m256	C41 = _mm256_setzero_ps();
            __m256	C42 = _mm256_setzero_ps();
            __m256	C43 = _mm256_setzero_ps();
            __m256	C44 = _mm256_setzero_ps();
            __m256	C45 = _mm256_setzero_ps();
            __m256	C46 = _mm256_setzero_ps();
            __m256	C47 = _mm256_setzero_ps();

            __m256	C50 = _mm256_setzero_ps();
            __m256	C51 = _mm256_setzero_ps();
            __m256	C52 = _mm256_setzero_ps();
            __m256	C53 = _mm256_setzero_ps();
            __m256	C54 = _mm256_setzero_ps();
            __m256	C55 = _mm256_setzero_ps();
            __m256	C56 = _mm256_setzero_ps();
            __m256	C57 = _mm256_setzero_ps();

            __m256	C60 = _mm256_setzero_ps();
            __m256	C61 = _mm256_setzero_ps();
            __m256	C62 = _mm256_setzero_ps();
            __m256	C63 = _mm256_setzero_ps();
            __m256	C64 = _mm256_setzero_ps();
            __m256	C65 = _mm256_setzero_ps();
            __m256	C66 = _mm256_setzero_ps();
            __m256	C67 = _mm256_setzero_ps();

            __m256	C70 = _mm256_setzero_ps();
            __m256	C71 = _mm256_setzero_ps();
            __m256	C72 = _mm256_setzero_ps();
            __m256	C73 = _mm256_setzero_ps();
            __m256	C74 = _mm256_setzero_ps();
            __m256	C75 = _mm256_setzero_ps();
            __m256	C76 = _mm256_setzero_ps();
            __m256	C77 = _mm256_setzero_ps();
            for (int l = 0; l < dim; l += 8) {
                C00 = _mm256_add_ps(C00, _mm256_mul_ps((*pA0),(*pB0)));
                C01 = _mm256_add_ps(C01, _mm256_mul_ps((*pA0),(*pB1)));
                C02 = _mm256_add_ps(C02, _mm256_mul_ps((*pA0),(*pB2)));
                C03 = _mm256_add_ps(C03, _mm256_mul_ps((*pA0),(*pB3)));
                C04 = _mm256_add_ps(C04, _mm256_mul_ps((*pA0),(*pB4)));
                C05 = _mm256_add_ps(C05, _mm256_mul_ps((*pA0),(*pB5)));
                C06 = _mm256_add_ps(C06, _mm256_mul_ps((*pA0),(*pB6)));
                C07 = _mm256_add_ps(C07, _mm256_mul_ps((*pA0),(*pB7)));

                C10 = _mm256_add_ps(C10, _mm256_mul_ps((*pA1),(*pB0)));
                C11 = _mm256_add_ps(C11, _mm256_mul_ps((*pA1),(*pB1)));
                C12 = _mm256_add_ps(C12, _mm256_mul_ps((*pA1),(*pB2)));
                C13 = _mm256_add_ps(C13, _mm256_mul_ps((*pA1),(*pB3)));
                C14 = _mm256_add_ps(C14, _mm256_mul_ps((*pA1),(*pB4)));
                C15 = _mm256_add_ps(C15, _mm256_mul_ps((*pA1),(*pB5)));
                C16 = _mm256_add_ps(C16, _mm256_mul_ps((*pA1),(*pB6)));
                C17 = _mm256_add_ps(C17, _mm256_mul_ps((*pA1),(*pB7)));

                C20 = _mm256_add_ps(C20, _mm256_mul_ps((*pA2),(*pB0)));
                C21 = _mm256_add_ps(C21, _mm256_mul_ps((*pA2),(*pB1)));
                C22 = _mm256_add_ps(C22, _mm256_mul_ps((*pA2),(*pB2)));
                C23 = _mm256_add_ps(C23, _mm256_mul_ps((*pA2),(*pB3)));
                C24 = _mm256_add_ps(C24, _mm256_mul_ps((*pA2),(*pB4)));
                C25 = _mm256_add_ps(C25, _mm256_mul_ps((*pA2),(*pB5)));
                C26 = _mm256_add_ps(C26, _mm256_mul_ps((*pA2),(*pB6)));
                C27 = _mm256_add_ps(C27, _mm256_mul_ps((*pA2),(*pB7)));

                C30 = _mm256_add_ps(C30, _mm256_mul_ps((*pA3),(*pB0)));
                C31 = _mm256_add_ps(C31, _mm256_mul_ps((*pA3),(*pB1)));
                C32 = _mm256_add_ps(C32, _mm256_mul_ps((*pA3),(*pB2)));
                C33 = _mm256_add_ps(C33, _mm256_mul_ps((*pA3),(*pB3)));
                C34 = _mm256_add_ps(C34, _mm256_mul_ps((*pA3),(*pB4)));
                C35 = _mm256_add_ps(C35, _mm256_mul_ps((*pA3),(*pB5)));
                C36 = _mm256_add_ps(C36, _mm256_mul_ps((*pA3),(*pB6)));
                C37 = _mm256_add_ps(C37, _mm256_mul_ps((*pA3),(*pB7)));

                C40 = _mm256_add_ps(C40, _mm256_mul_ps((*pA4),(*pB0)));
                C41 = _mm256_add_ps(C41, _mm256_mul_ps((*pA4),(*pB1)));
                C42 = _mm256_add_ps(C42, _mm256_mul_ps((*pA4),(*pB2)));
                C43 = _mm256_add_ps(C43, _mm256_mul_ps((*pA4),(*pB3)));
                C44 = _mm256_add_ps(C44, _mm256_mul_ps((*pA4),(*pB4)));
                C45 = _mm256_add_ps(C45, _mm256_mul_ps((*pA4),(*pB5)));
                C46 = _mm256_add_ps(C46, _mm256_mul_ps((*pA4),(*pB6)));
                C47 = _mm256_add_ps(C47, _mm256_mul_ps((*pA4),(*pB7)));

                C50 = _mm256_add_ps(C50, _mm256_mul_ps((*pA5),(*pB0)));
                C51 = _mm256_add_ps(C51, _mm256_mul_ps((*pA5),(*pB1)));
                C52 = _mm256_add_ps(C52, _mm256_mul_ps((*pA5),(*pB2)));
                C53 = _mm256_add_ps(C53, _mm256_mul_ps((*pA5),(*pB3)));
                C54 = _mm256_add_ps(C54, _mm256_mul_ps((*pA5),(*pB4)));
                C55 = _mm256_add_ps(C55, _mm256_mul_ps((*pA5),(*pB5)));
                C56 = _mm256_add_ps(C56, _mm256_mul_ps((*pA5),(*pB6)));
                C57 = _mm256_add_ps(C57, _mm256_mul_ps((*pA5),(*pB7)));

                C60 = _mm256_add_ps(C60, _mm256_mul_ps((*pA6),(*pB0)));
                C61 = _mm256_add_ps(C61, _mm256_mul_ps((*pA6),(*pB1)));
                C62 = _mm256_add_ps(C62, _mm256_mul_ps((*pA6),(*pB2)));
                C63 = _mm256_add_ps(C63, _mm256_mul_ps((*pA6),(*pB3)));
                C64 = _mm256_add_ps(C64, _mm256_mul_ps((*pA6),(*pB4)));
                C65 = _mm256_add_ps(C65, _mm256_mul_ps((*pA6),(*pB5)));
                C66 = _mm256_add_ps(C66, _mm256_mul_ps((*pA6),(*pB6)));
                C67 = _mm256_add_ps(C67, _mm256_mul_ps((*pA6),(*pB7)));

                C70 = _mm256_add_ps(C70, _mm256_mul_ps((*pA7),(*pB0)));
                C71 = _mm256_add_ps(C71, _mm256_mul_ps((*pA7),(*pB1)));
                C72 = _mm256_add_ps(C72, _mm256_mul_ps((*pA7),(*pB2)));
                C73 = _mm256_add_ps(C73, _mm256_mul_ps((*pA7),(*pB3)));
                C74 = _mm256_add_ps(C74, _mm256_mul_ps((*pA7),(*pB4)));
                C75 = _mm256_add_ps(C75, _mm256_mul_ps((*pA7),(*pB5)));
                C76 = _mm256_add_ps(C76, _mm256_mul_ps((*pA7),(*pB6)));
                C77 = _mm256_add_ps(C77, _mm256_mul_ps((*pA7),(*pB7)));

                pA0++;
                pA1++;
                pA2++;
                pA3++;
                pA4++;
                pA5++;
                pA6++;
                pA7++;
                pB0++;
                pB1++;
                pB2++;
                pB3++;
                pB4++;
                pB5++;
                pB6++;
                pB7++;
            }


            __m256 sumab = _mm256_hadd_ps(C00, C01);
            __m256 sumcd = _mm256_hadd_ps(C02, C03);
            __m256 sumef = _mm256_hadd_ps(C04, C05);
            __m256 sumgh = _mm256_hadd_ps(C06, C07);

            __m256 sum1 = _mm256_hadd_ps(sumab, sumcd);
            __m256 sum2 = _mm256_hadd_ps(sumef, sumgh);

            __m256 blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            __m256 perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            __m256 sum = _mm256_add_ps(perm, blend);

            _mm256_store_ps(&C.data[(n + C.oRows) * N + (m + C.oCols)], sum);


            sumab = _mm256_hadd_ps(C10, C11);
            sumcd = _mm256_hadd_ps(C12, C13);
            sumef = _mm256_hadd_ps(C14, C15);
            sumgh = _mm256_hadd_ps(C16, C17);

            sum1 = _mm256_hadd_ps(sumab, sumcd);
            sum2 = _mm256_hadd_ps(sumef, sumgh);

            blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            sum = _mm256_add_ps(perm, blend);

            _mm256_store_ps(&C.data[(n + 1 + C.oRows) * N + (m + C.oCols)], sum);

            sumab = _mm256_hadd_ps(C20, C21);
            sumcd = _mm256_hadd_ps(C22, C23);
            sumef = _mm256_hadd_ps(C24, C25);
            sumgh = _mm256_hadd_ps(C26, C27);

            sum1 = _mm256_hadd_ps(sumab, sumcd);
            sum2 = _mm256_hadd_ps(sumef, sumgh);

            blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            sum = _mm256_add_ps(perm, blend);
            _mm256_store_ps(&C.data[(n + 2 + C.oRows) * N + (m + C.oCols)], sum);

            sumab = _mm256_hadd_ps(C30, C31);
            sumcd = _mm256_hadd_ps(C32, C33);
            sumef = _mm256_hadd_ps(C34, C35);
            sumgh = _mm256_hadd_ps(C36, C37);

            sum1 = _mm256_hadd_ps(sumab, sumcd);
            sum2 = _mm256_hadd_ps(sumef, sumgh);

            blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            sum = _mm256_add_ps(perm, blend);

            _mm256_store_ps(&C.data[(n + 3 + C.oRows) * N + (m + C.oCols)], sum);

            sumab = _mm256_hadd_ps(C40, C41);
            sumcd = _mm256_hadd_ps(C42, C43);
            sumef = _mm256_hadd_ps(C44, C45);
            sumgh = _mm256_hadd_ps(C46, C47);

            sum1 = _mm256_hadd_ps(sumab, sumcd);
            sum2 = _mm256_hadd_ps(sumef, sumgh);

            blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            sum = _mm256_add_ps(perm, blend);

            _mm256_store_ps(&C.data[(n + 4 + C.oRows) * N + (m + C.oCols)], sum);

            sumab = _mm256_hadd_ps(C50, C51);
            sumcd = _mm256_hadd_ps(C52, C53);
            sumef = _mm256_hadd_ps(C54, C55);
            sumgh = _mm256_hadd_ps(C56, C57);

            sum1 = _mm256_hadd_ps(sumab, sumcd);
            sum2 = _mm256_hadd_ps(sumef, sumgh);

            blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            sum = _mm256_add_ps(perm, blend);


            _mm256_store_ps(&C.data[(n + 5 + C.oRows) * N + (m + C.oCols)], sum);

            sumab = _mm256_hadd_ps(C60, C61);
            sumcd = _mm256_hadd_ps(C62, C63);
            sumef = _mm256_hadd_ps(C64, C65);
            sumgh = _mm256_hadd_ps(C66, C67);

            sum1 = _mm256_hadd_ps(sumab, sumcd);
            sum2 = _mm256_hadd_ps(sumef, sumgh);

            blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            sum = _mm256_add_ps(perm, blend);

            _mm256_store_ps(&C.data[(n + 6 + C.oRows) * N + (m + C.oCols)], sum);

            sumab = _mm256_hadd_ps(C70, C71);
            sumcd = _mm256_hadd_ps(C72, C73);
            sumef = _mm256_hadd_ps(C74, C75);
            sumgh = _mm256_hadd_ps(C76, C77);

            sum1 = _mm256_hadd_ps(sumab, sumcd);
            sum2 = _mm256_hadd_ps(sumef, sumgh);

            blend = _mm256_blend_ps(sum1, sum2, 0xf0);
            perm = _mm256_permute2f128_ps(sum1, sum2, 0x21);
            sum = _mm256_add_ps(perm, blend);

            _mm256_store_ps(&C.data[(n + 7 + C.oRows) * N + (m + C.oCols)], sum);
        }
    }
}

inline
void strassen(Matrix& A, Matrix& B, Matrix& C, Matrix& P, Matrix& Ps, Matrix& S, Matrix& T) {
    int	dim = A.getDim();
    int	dim2 = dim - dim / 2;

    Matrix	A1 = A.getSubMatrix(0, 0, dim2);
    Matrix	A2 = A.getSubMatrix(0, dim2, dim2);
    Matrix	A3 = A.getSubMatrix(dim2, 0, dim2);
    Matrix	A4 = A.getSubMatrix(dim2, dim2, dim2);

    Matrix	B1 = B.getSubMatrix(0, 0, dim2);
    Matrix	B2 = B.getSubMatrix(0, dim2, dim2);
    Matrix	B3 = B.getSubMatrix(dim2, 0, dim2);
    Matrix	B4 = B.getSubMatrix(dim2, dim2, dim2);

    Matrix	C1 = C.getSubMatrix(0, 0, dim2);
    Matrix	C2 = C.getSubMatrix(0, dim2, dim2);
    Matrix	C3 = C.getSubMatrix(dim2, 0, dim2);
    Matrix	C4 = C.getSubMatrix(dim2, dim2, dim2);

    Matrix	S1 = S.getSubMatrix(0, 0, dim2);
    Matrix	S2 = S.getSubMatrix(0, dim2, dim2);
    Matrix	S3 = S.getSubMatrix(dim2, 0, dim2);
    Matrix	S4 = S.getSubMatrix(dim2, dim2, dim2);

    Matrix	T1 = T.getSubMatrix(0, 0, dim2);
    Matrix	T2 = T.getSubMatrix(0, dim2, dim2);
    Matrix	T3 = T.getSubMatrix(dim2, 0, dim2);
    Matrix	T4 = T.getSubMatrix(dim2, dim2, dim2);

    Matrix	P1 = P.getSubMatrix(0, 0, dim2);
    Matrix	P2 = P.getSubMatrix(0, dim2, dim2);
    Matrix	P3 = P.getSubMatrix(dim2, 0, dim2);
    Matrix	P4 = P.getSubMatrix(dim2, dim2, dim2);

    Matrix	P5 = Ps.getSubMatrix(0, 0, dim2);
    Matrix	P6 = Ps.getSubMatrix(0, dim2, dim2);
    Matrix	P7 = Ps.getSubMatrix(dim2, 0, dim2);

    for (int i = 0; i < dim2; ++i) {
        for (int j = 0; j < dim2; j += 8) {
            __m256*	pA1 = A1.get(i, j);
            __m256*	pA2 = A2.get(i, j);
            __m256*	pA3 = A3.get(i, j);
            __m256*	pA4 = A4.get(i, j);
            __m256*	pS2 = S2.get(i, j);
            S1.set(i, j, _mm256_add_ps((*pA3), (*pA4)));
            S2.set(i, j, _mm256_add_ps((*pA3), _mm256_sub_ps((*pA4),(*pA1))));
            S3.set(i, j, _mm256_sub_ps((*pA1), (*pA3)));
            S4.set(i, j, _mm256_sub_ps((*pA2), (*pS2)));
            pA1++;
            pA2++;
            pA3++;
            pA4++;
            pS2++;

            __m256*	pB1 = B1.getT(j, i);
            __m256*	pB2 = B2.getT(j, i);
            __m256*	pB3 = B3.getT(j, i);
            __m256*	pB4 = B4.getT(j, i);
            __m256*	pT2 = T2.getT(j, i);
            T1.setT(j, i, _mm256_sub_ps((*pB2),(*pB1)));
            T2.setT(j, i, _mm256_sub_ps((*pB4), (_mm256_sub_ps((*pB2), (*pB1)))));
            T3.setT(j, i, _mm256_sub_ps((*pB4), (*pB2)));
            T4.setT(j, i, _mm256_sub_ps((*pB3), (*pT2)));
            pB1++;
            pB2++;
            pB3++;
            pB4++;
            pT2++;
        }
    }



    if (dim2 < TRUNCATION_POINT) {
        naive(A1, B1, P1);
        naive(A2, B3, P2);
        naive(S1, T1, P3);
        naive(S2, T2, P4);
        naive(S3, T3, P5);
        naive(S4, B4, P6);
        naive(A4, T4, P7);

    } else {
        strassen(A1, B1, P1, C1, C2, C3, C4);
        strassen(A2, B3, P2, C1, C2, C3, C4);
        strassen(S1, T1, P3, C1, C2, C3, C4);
        strassen(S2, T2, P4, C1, C2, C3, C4);
        strassen(S3, T3, P5, C1, C2, C3, C4);
        strassen(S4, B4, P6, C1, C2, C3, C4);
        strassen(A4, T4, P7, C1, C2, C3, C4);
    }

    for (int i=0; i < dim2; ++i) {
        for (int j = 0; j < dim2; j += 8){
            __m256*	pP1 = P1.get(i, j);
            __m256*	pP2 = P2.get(i, j);
            __m256*	pP3 = P3.get(i, j);
            __m256*	pP4 = P4.get(i, j);
            __m256*	pP5 = P5.get(i, j);
            __m256*	pP6 = P6.get(i, j);
            __m256*	pP7 = P7.get(i, j);
            C1.set(i, j, _mm256_add_ps((*pP1), (*pP2)));
            C3.set(i, j, _mm256_add_ps((*pP1), _mm256_add_ps((*pP4), _mm256_add_ps((*pP5),(*pP7)))));
            C4.set(i, j, _mm256_add_ps((*pP1), _mm256_add_ps((*pP4), _mm256_add_ps((*pP5),(*pP3)))));
            C2.set(i, j, _mm256_add_ps((*pP1), _mm256_add_ps((*pP4), _mm256_add_ps((*pP3),(*pP6)))));
            pP1++;
            pP2++;
            pP3++;
            pP4++;
            pP5++;
            pP6++;
            pP7++;
        }
    }
}


void multiply(float* mat1, float* mat2, int N, float* res) {
    float* dat4 = (float*)aligned_alloc(ALIGNMENT, sizeof(float) * N * N);
    float* dat5 = (float*)aligned_alloc(ALIGNMENT, sizeof(float) * N * N);
    float* dat6 = (float*)aligned_alloc(ALIGNMENT, sizeof(float) * N * N);
    float* dat7 = (float*)aligned_alloc(ALIGNMENT, sizeof(float) * N * N);

    Matrix A(mat1, mat1, N, 0, 0);
    Matrix B(mat2, mat2, N, 0, 0);
    Matrix C(res, res, N, 0, 0);
    Matrix P(dat4, dat4, N, 0, 0);
    Matrix Ps(dat5, dat5, N, 0, 0);
    Matrix S(dat6, dat6, N, 0, 0);
    Matrix T(dat7, dat7, N, 0, 0);

    TR(B);
    strassen(A, B, C, P, Ps, S, T);
}

int main() {
    float* mat1 = (float*)aligned_alloc(ALIGNMENT, sizeof(float) * N * N);
    float* mat2 = (float*)aligned_alloc(ALIGNMENT, sizeof(float) * N * N);
    float* res = (float*)aligned_alloc(ALIGNMENT, sizeof(float) * N * N);

    for (int i = 0; i < N * N; ++i) {
        mat1[i] = rand() % 2;
        mat2[i] = rand() % 2;
    }

    time_t time = clock();
    multiply(mat1, mat2, N, res);
    std::cout << "Time: " << (double)(clock() - time) / CLOCKS_PER_SEC;
    return 0;
}
