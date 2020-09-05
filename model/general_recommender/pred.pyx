# tag: numpy_old
# You can ignore the previous line.
# It's for internal testing of the cython documentation.
cimport cython
#The new code after disabling such features is as follows.
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float32
INT_DTYPE = np.long
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t
ctypedef np.int_t INT_DTYPE_t
from cpython cimport dict, array
import array
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
#
# The arrays f, g and h is typed as "np.ndarray" instances. The only effect
# this has is to a) insert checks that the function arguments really are
# NumPy arrays, and b) make some attribute access like f.shape[0] much
# more efficient. (In this example this doesn't matter though.)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def pred(np.ndarray[INT_DTYPE_t, ndim=1] user_ids, dict train_dict, np.ndarray[DTYPE_t, ndim=2] cur_embedding_Q_,
                   np.ndarray[DTYPE_t, ndim=2] cur_embedding_Q, np.ndarray[DTYPE_t, ndim=1] cur_bias, np.ndarray[DTYPE_t, ndim=2] cur_W,
                   np.ndarray[DTYPE_t, ndim=2] cur_h, np.ndarray[DTYPE_t, ndim=2] cur_b, int num_user, int num_item, int alg, int act, float beta):
    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    #
    # For the indices, the "int" type is used. This corresponds to a C int,
    # other C types (like "unsigned int") could have been used instead.
    # Purists could use "Py_ssize_t" which is the proper Python type for
    # array indices.
    cdef int u, num_idx
    cdef np.ndarray cand_items_by_u
    cdef np.ndarray[DTYPE_t, ndim=2] embedding_q_
    cdef np.ndarray embedding_q
    cdef np.ndarray ratings=np.zeros((num_user, num_item), dtype=DTYPE)
    cdef np.ndarray q_
    cdef int b,n,r
    cdef np.ndarray[DTYPE_t, ndim=2] mlp_output
    cdef np.ndarray[DTYPE_t, ndim=2] A_
    cdef np.ndarray[DTYPE_t, ndim=2] exp_A_
    cdef np.ndarray[DTYPE_t, ndim=2] exp_sum
    cdef np.ndarray[DTYPE_t, ndim=3] A
    cdef np.ndarray[DTYPE_t, ndim=2] embedding_p
    cdef np.ndarray[DTYPE_t, ndim=1] output
    for u in user_ids:
        print(u)
        cand_items_by_u = np.array(train_dict[u])
        num_idx = len(cand_items_by_u)
        #item_idx = np.full(self.num_items, num_idx, dtype=np.int32)
        #user_input.extend([cand_items_by_u]*self.num_items)
        embedding_q_ = cur_embedding_Q_[cand_items_by_u] # Q q_j
        embedding_q = np.expand_dims(cur_embedding_Q, 1) #p_i
        q_ = embedding_q_ * embedding_q
        b = np.shape(q_)[0]
        n = np.shape(q_)[1]
        r = (alg + 1) * 64

        mlp_output = np.matmul(np.reshape(q_, [-1, r]), cur_W) + cur_b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
        if act == 0:
            mlp_output = np.maximum(0, mlp_output)
        elif act == 1:
            mlp_output = 1 / (1 + np.exp(-mlp_output))
        elif act == 2:
            mlp_output = np.tanh(mlp_output)

        A_ = np.reshape(np.matmul(mlp_output, cur_h), [b, n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

        # softmax for not mask features
        exp_A_ = np.exp(A_)
        # mask_mat 标志每个用户的商品，为了计算不同长度的sum
        exp_sum = np.sum(exp_A_, axis=1, keepdims=True)  # (b, 1)
        exp_sum = np.power(exp_sum, beta)

        A = np.expand_dims(np.divide(exp_A_, exp_sum), 2)  # (b, n, 1)
        embedding_q = np.sum(embedding_q, 1)
        embedding_p = np.sum(A * embedding_q_, 1)
        output = np.sum(embedding_p * embedding_q, 1) + cur_bias
        ratings[u] = output
    return ratings

def add(int a, int b):
    return a + b