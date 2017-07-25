Help on function Convolution in module mxnet.symbol:

Convolution(*args, **kwargs)
    Compute *N*-D convolution on *(N+2)*-D input.

    In the 2-D convolution, given input data with shape *(batch_size,
    channel, height, width)*, the output is computed by

    .. math::

       out[n,i,:,:] = bias[i] + \sum_{j=0}^{num\_filter} data[n,j,:,:] \star
       weight[i,j,:,:]

    where :math:`\star` is the 2-D cross-correlation operator.

    For general 2-D convolution, the shapes are

    - **data**: *(batch_size, channel, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*.

    Define::

      f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

    then we have::

      out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
      out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

    If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

    The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
    width)*. We can choose other layouts such as *NHWC*.

    If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
    evenly into *g* parts along the channel axis, and also evenly split ``weight``
    along the first dimension. Next compute the convolution on the *i*-th part of
    the data with the *i*-th weight part. The output is obtained by concatenating all
    the *g* results.

    1-D convolution does not have *height* dimension but only *width* in space.

    - **data**: *(batch_size, channel, width)*
    - **weight**: *(num_filter, channel, kernel[0])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_width)*.

    3-D convolution adds an additional *depth* dimension besides *height* and
    *width*. The shapes are

    - **data**: *(batch_size, channel, depth, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.

    Both ``weight`` and ``bias`` are learnable parameters.

    There are other options to tune the performance.

    - **cudnn_tune**: enable this option leads to higher startup time but may give
      faster speed. Options are

      - **off**: no tuning
      - **limited_workspace**:run test and pick the fastest algorithm that doesn't
        exceed workspace limit.
    - **fastest**: pick the fastest algorithm and ignore workspace limit.
      - **None** (default): the behavior is determined by environment variable
        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
        (default), 2 for fastest.

    - **workspace**: A large number leads to more (GPU) memory usage but may improve
      the performance.



    Defined in src/operator/convolution.cc:L154

    Parameters
    ----------
    data : Symbol
        Input data to the ConvolutionOp.
    weight : Symbol
        Weight matrix.
    bias : Symbol
        Bias parameter.
    kernel : Shape(tuple), required
        convolution kernel size: (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=()
        convolution stride: (h, w) or (d, h, w)
    dilate : Shape(tuple), optional, default=()
        convolution dilate: (h, w) or (d, h, w)
   pad : Shape(tuple), optional, default=()
        pad for convolution: (h, w) or (d, h, w)
    num_filter : int (non-negative), required
        convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions.
    workspace : long (non-negative), optional, default=1024
        Maximum temporary workspace allowed for convolution (MB).
    no_bias : boolean, optional, default=False
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algo by running performance test.
    cudnn_off : boolean, optional, default=False
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.



==========
Help on function Activation in module mxnet.symbol:

Activation(*args, **kwargs)
    Applies an activation function element-wise to the input.

    The following activation functions are supported:

    - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
    - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
    - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
    - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`



    Defined in src/operator/activation.cc:L77

    Parameters
    ----------
    data : Symbol
        Input array to activation function.
    act_type : {'relu', 'sigmoid', 'softrelu', 'tanh'}, required
        Activation function to be applied.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.

    Examples
    --------
    A one-hidden-layer MLP with ReLU activation:

    >>> data = Variable('data')
    >>> mlp = FullyConnected(data=data, num_hidden=128, name='proj')
    >>> mlp = Activation(data=mlp, act_type='relu', name='activation')
    >>> mlp = FullyConnected(data=mlp, num_hidden=10, name='mlp')
    >>> mlp
    <Symbol mlp>

    Regression Test
    ---------------
    ReLU activation

    >>> test_suites = [
    ... ('relu', lambda x: np.maximum(x, 0)),
    ... ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
    ... ('tanh', lambda x: np.tanh(x)),
    ... ('softrelu', lambda x: np.log(1 + np.exp(x)))
    ... ]
    >>> x = test_utils.random_arrays((2, 3, 4))
    >>> for act_type, numpy_impl in test_suites:
    ... op = Activation(act_type=act_type, name='act')
    ... y = test_utils.simple_forward(op, act_data=x)
    ... y_np = numpy_impl(x)
    ... print('%s: %s' % (act_type, test_utils.almost_equal(y, y_np)))
    relu: True
    sigmoid: True
    tanh: True
    softrelu: True

==========

Help on function BatchNorm in module mxnet.symbol:

BatchNorm(*args, **kwargs)
    Batch normalization.

    Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.

    Assume the input has more than one dimension and we normalize along axis 1.
    We first compute the mean and variance along this axis:

    .. math::

      data\_mean[i] = mean(data[:,i,:,...]) \\
      data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

      out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
    ``data_var`` as well, which are needed for the backward pass.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated
    by::

      moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
      moving_var = moving_var * momentum + data_var * (1 - momentum)

    If ``use_global_stats`` is set to be true, then ``moving_mean`` and
    ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
    the output. It is often used during inference.

    Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
    then set ``gamma`` to 1 and its gradient to 0.



    Defined in src/operator/batch_norm.cc:L523

    Parameters
    ----------
    data : Symbol
        Input data to batch normalization
    gamma : Symbol
        gamma array
    beta : Symbol
        beta array
    moving_mean : Symbol
        running mean of input
    moving_var : Symbol
        running variance of input
    eps : float, optional, default=0.001
        Epsilon to prevent div 0. Must be bigger than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)
    momentum : float, optional, default=0.9
        Momentum for moving average
    fix_gamma : boolean, optional, default=True
        Fix gamma while training
    use_global_stats : boolean, optional, default=False
        Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.
    output_mean_var : boolean, optional, default=False
        Output All,normal mean and var
    cudnn_off : boolean, optional, default=False
        Do not select CUDNN operator, if available

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.


==========
Help on function Pooling in module mxnet.symbol:

Pooling(*args, **kwargs)
    Performs pooling on the input.

    The shapes for 1-D pooling are

    - **data**: *(batch_size, channel, width)*,
    - **out**: *(batch_size, num_filter, out_width)*.

    The shapes for 2-D pooling are

    - **data**: *(batch_size, channel, height, width)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*, with::

        out_height = f(height, kernel[0], pad[0], stride[0])
        out_width = f(width, kernel[1], pad[1], stride[1])

    The definition of *f* depends on ``pooling_convention``, which has two options:

    - **valid** (default)::

        f(x, k, p, s) = floor((x+2*p-k)/s)+1

    - **full**, which is compatible with Caffe::

        f(x, k, p, s) = ceil((x+2*p-k)/s)+1

    But ``global_pool`` is set to be true, then do a global pooling, namely reset
    ``kernel=(height, width)``.

    Three pooling options are supported by ``pool_type``:

    - **avg**: average pooling
    - **max**: max pooling
    - **sum**: sum pooling

    For 3-D pooling, an additional *depth* dimension is added before
    *height*. Namely the input data will have shape *(batch_size, channel, depth,
    height, width)*.



    Defined in src/operator/pooling.cc:L121

    Parameters
    ----------
    data : Symbol
        Input data to the pooling operator.
    global_pool : boolean, optional, default=False
        Ignore kernel size, do global pooling based on current input feature map.
    cudnn_off : boolean, optional, default=False
        Turn off cudnn pooling and use MXNet pooling operator.
    kernel : Shape(tuple), required
        pooling kernel size: (y, x) or (d, y, x)
    pool_type : {'avg', 'max', 'sum'}, required
        Pooling type to be applied.
    pooling_convention : {'full', 'valid'},optional, default='valid'
        Pooling convention to be applied.
    stride : Shape(tuple), optional, default=()
        stride: for pooling (y, x) or (d, y, x)
    pad : Shape(tuple), optional, default=()
        pad for pooling: (y, x) or (d, y, x)

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
