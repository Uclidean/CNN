import numpy as np
def conv_backward_naive(dout, cache):
      """
      A naive implementation of the backward pass for a convolutional layer.

      Inputs:
      - dout: Upstream derivatives.
      - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

      Returns a tuple of:
      - dx: Gradient with respect to x
      - dw: Gradient with respect to w
      - db: Gradient with respect to b
      """
      (x, w, b, conv_param) = cache
      pad = conv_param['pad'] # Padding number
      stride = conv_param['stride'] # Stride number
      input_N, input_c, input_h, input_w = x.shape # Input features
      x_padded = np.pad(x, ((0,0),(0,0),(pad,pad), # Input with padding applied
                        (pad,pad)), 'constant')
      N, i_c, i_h, i_w = x_padded.shape # Padded input features
      F, f_c, f_h, f_w = w.shape # Filter features
      db = np.zeros_like(b) # Derivative of bias terms
      dw = np.zeros_like(w) # Derivative of filters
      dx_pad = np.zeros_like(x_padded) # Derivative of input
      for n in range(N): # Iterate through all inputs
        for i in range(F): # Iterate through all filters
          db[i] += np.sum(dout[n, i]) #dL/db = dL/dO * 1
          y_ax = act_y = 0 # slide filter along vertical axis
          while (y_ax + f_h <= i_h): # filter is within vertical bounds
            x_ax = act_x = 0 # slide filter along horizontal axis
            while (x_ax + f_w <= i_w): # filter is within horizontal bounds
              dw[i] += dout[n, i, act_y, act_x] * x_padded[n, ::, y_ax:y_ax+f_h, x_ax:x_ax+f_w] #dL/dw = dL/dO * dO/dF = dL/dO * X
              dx_pad[n, ::, y_ax:y_ax+f_h, x_ax:x_ax+f_w] += dout[n, i, act_y, act_x] * w[i] #dL/dX = dL/dO * dO/dX = dL/dO * F
              x_ax += stride
              act_x += 1
            y_ax += stride
            act_y += 1
      dx = dx_pad[::, ::, pad:pad+input_h, pad:pad+input_w] # Remove padding
      return dx, dw, db


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    (x, pool_param) = cache
    N, C, H, W = x.shape # Input features
    pool_height = pool_param['pool_height'] # Height of pool
    pool_width = pool_param['pool_width'] # Width of pool
    stride = pool_param['stride'] # Stride number
    dx = np.zeros_like(x) # Derivative of input

    ## Derivatives of non-max values within pool-sized patch of image will be 0

    for n in range(N): # Iterate through all inputs
      for c in range(C): # Iterate through all columns
        out_y = curr_y = 0
        while (curr_y + pool_height <= H): # Slide filter vertically
          out_x = curr_x = 0
          while (curr_x + pool_width <= W): # Slide filter horizontally
            # Get indeces of max value within pool-sized patch of image
            ind = np.unravel_index(np.argmax(x[n, c, curr_y:curr_y + pool_height, curr_x:curr_x+pool_width]), (pool_height, pool_width))
            dx[n, c, curr_y:curr_y+pool_height, curr_x:curr_x+pool_width][ind] = dout[n, c, out_y, out_x]
            out_x +=1
            curr_x += stride
          out_y +=1
          curr_y += stride
    return dx
