??
?&?%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
;
Minimum
x"T
y"T
z"T"
Ttype:

2	?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(?
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
R
SoftplusGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2
?
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12b'v1.13.1-0-g6612da8951'??
n
PlaceholderPlaceholder*
dtype0*
shape:?????????<*'
_output_shapes
:?????????<
p
Placeholder_1Placeholder*
dtype0*
shape:?????????*'
_output_shapes
:?????????
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_3Placeholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
h
Placeholder_4Placeholder*
dtype0*
shape:?????????*#
_output_shapes
:?????????
h
Placeholder_5Placeholder*
dtype0*
shape:?????????*#
_output_shapes
:?????????
h
Placeholder_6Placeholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
h
Placeholder_7Placeholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
N
Placeholder_8Placeholder*
dtype0*
_output_shapes
: *
shape: 
N
Placeholder_9Placeholder*
shape: *
dtype0*
_output_shapes
: 
?
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"<   @   *"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
:
?
.pi/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *??a?*"
_class
loc:@pi/dense/kernel
?
.pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *??a>*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: 
?
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
_output_shapes

:<@*
dtype0*"
_class
loc:@pi/dense/kernel*

seed*
seed2*
T0
?
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@pi/dense/kernel
?
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:<@*"
_class
loc:@pi/dense/kernel*
T0
?
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
_output_shapes

:<@*
T0
?
pi/dense/kernel
VariableV2*
	container *
_output_shapes

:<@*
shared_name *
dtype0*"
_class
loc:@pi/dense/kernel*
shape
:<@
?
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
_output_shapes

:<@*
validate_shape(
~
pi/dense/kernel/readIdentitypi/dense/kernel*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:<@
?
pi/dense/bias/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB@*    *
_output_shapes
:@*
dtype0
?
pi/dense/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name * 
_class
loc:@pi/dense/bias*
dtype0
?
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
t
pi/dense/bias/readIdentitypi/dense/bias*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0
?
pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*'
_output_shapes
:?????????@*
T0*
transpose_b( *
transpose_a( 
?
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*'
_output_shapes
:?????????@*
T0*
data_formatNHWC
Y
pi/dense/TanhTanhpi/dense/BiasAdd*
T0*'
_output_shapes
:?????????@
?
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
?
0pi/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
dtype0*
valueB
 *׳]?
?
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB
 *׳]>*
_output_shapes
: 
?
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*

seed*
T0*$
_class
loc:@pi/dense_1/kernel*
dtype0*
seed2*
_output_shapes

:@@
?
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: 
?
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
?
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@
?
pi/dense_1/kernel
VariableV2*$
_class
loc:@pi/dense_1/kernel*
shared_name *
shape
:@@*
_output_shapes

:@@*
	container *
dtype0
?
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes

:@@*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
?
pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
?
!pi/dense_1/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
valueB@*    *
dtype0
?
pi/dense_1/bias
VariableV2*
	container *
dtype0*
shape:@*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
shared_name 
?
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
z
pi/dense_1/bias/readIdentitypi/dense_1/bias*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
T0
?
pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
T0*'
_output_shapes
:?????????@*
transpose_b( *
transpose_a( 
?
pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:?????????@*
T0
]
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????@
?
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *
_output_shapes
:*$
_class
loc:@pi/dense_2/kernel*
dtype0
?
0pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *?_??*$
_class
loc:@pi/dense_2/kernel*
dtype0
?
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
valueB
 *?_?>*
dtype0
?
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*

seed*
dtype0*
seed2/*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
?
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: *
T0
?
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
?
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
?
pi/dense_2/kernel
VariableV2*
shared_name *
shape
:@*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
dtype0*
	container 
?
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
_output_shapes

:@*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
?
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel
?
!pi/dense_2/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
valueB*    *
dtype0
?
pi/dense_2/bias
VariableV2*
shape:*"
_class
loc:@pi/dense_2/bias*
dtype0*
	container *
_output_shapes
:*
shared_name 
?
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
?
pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*'
_output_shapes
:?????????*
transpose_b( *
transpose_a( *
T0
?
pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:?????????*
T0
i
pi/log_std/initial_valueConst*
_output_shapes
:*
dtype0*
valueB"   ?   ?
v

pi/log_std
VariableV2*
_output_shapes
:*
shared_name *
dtype0*
	container *
shape:
?
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*
_class
loc:@pi/log_std
k
pi/log_std/readIdentity
pi/log_std*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
C
pi/ExpExppi/log_std/read*
_output_shapes
:*
T0
Z
pi/ShapeShapepi/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
Z
pi/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
pi/random_normal/stddevConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*
T0*
dtype0*'
_output_shapes
:?????????*
seed2D*

seed
?
pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*'
_output_shapes
:?????????*
T0
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*
T0*'
_output_shapes
:?????????
Y
pi/mulMulpi/random_normalpi/Exp*'
_output_shapes
:?????????*
T0
[
pi/addAddpi/dense_2/BiasAddpi/mul*'
_output_shapes
:?????????*
T0
b
pi/subSubPlaceholder_1pi/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
E
pi/Exp_1Exppi/log_std/read*
_output_shapes
:*
T0
O

pi/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2
J
pi/add_1Addpi/Exp_1
pi/add_1/y*
T0*
_output_shapes
:
Y

pi/truedivRealDivpi/subpi/add_1*
T0*'
_output_shapes
:?????????
M
pi/pow/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
U
pi/powPow
pi/truedivpi/pow/y*'
_output_shapes
:?????????*
T0
O

pi/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
T0*
_output_shapes
:
S
pi/add_2Addpi/powpi/mul_1*
T0*'
_output_shapes
:?????????
O

pi/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *????
W
pi/add_3Addpi/add_2
pi/add_3/y*
T0*'
_output_shapes
:?????????
O

pi/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*
T0*'
_output_shapes
:?????????
Z
pi/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( *#
_output_shapes
:?????????
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
E
pi/Exp_2Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_4/yConst*
dtype0*
valueB
 *w?+2*
_output_shapes
: 
J
pi/add_4Addpi/Exp_2
pi/add_4/y*
T0*
_output_shapes
:
]
pi/truediv_1RealDivpi/sub_1pi/add_4*
T0*'
_output_shapes
:?????????
O

pi/pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*
T0*'
_output_shapes
:?????????
O

pi/mul_3/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
T0*
_output_shapes
:
U
pi/add_5Addpi/pow_1pi/mul_3*
T0*'
_output_shapes
:?????????
O

pi/add_6/yConst*
_output_shapes
: *
valueB
 *????*
dtype0
W
pi/add_6Addpi/add_5
pi/add_6/y*'
_output_shapes
:?????????*
T0
O

pi/mul_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*
T0*'
_output_shapes
:?????????
\
pi/Sum_1/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
?
pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:?????????*

Tidx0
q
pi/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
s
pi/Placeholder_1Placeholder*
shape:?????????*'
_output_shapes
:?????????*
dtype0
O

pi/mul_5/xConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
T0*
_output_shapes
:
>
pi/Exp_3Exppi/mul_5*
_output_shapes
:*
T0
O

pi/mul_6/xConst*
dtype0*
_output_shapes
: *
valueB
 *   @
_
pi/mul_6Mul
pi/mul_6/xpi/Placeholder_1*
T0*'
_output_shapes
:?????????
K
pi/Exp_4Exppi/mul_6*'
_output_shapes
:?????????*
T0
e
pi/sub_2Subpi/Placeholderpi/dense_2/BiasAdd*'
_output_shapes
:?????????*
T0
O

pi/pow_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
W
pi/pow_2Powpi/sub_2
pi/pow_2/y*
T0*'
_output_shapes
:?????????
U
pi/add_7Addpi/pow_2pi/Exp_3*'
_output_shapes
:?????????*
T0
O

pi/add_8/yConst*
valueB
 *w?+2*
_output_shapes
: *
dtype0
W
pi/add_8Addpi/Exp_4
pi/add_8/y*
T0*'
_output_shapes
:?????????
]
pi/truediv_2RealDivpi/add_7pi/add_8*
T0*'
_output_shapes
:?????????
O

pi/sub_3/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
[
pi/sub_3Subpi/truediv_2
pi/sub_3/y*
T0*'
_output_shapes
:?????????
O

pi/mul_7/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
W
pi/mul_7Mul
pi/mul_7/xpi/sub_3*
T0*'
_output_shapes
:?????????
]
pi/add_9Addpi/mul_7pi/Placeholder_1*
T0*'
_output_shapes
:?????????
\
pi/sub_4Subpi/add_9pi/log_std/read*'
_output_shapes
:?????????*
T0
\
pi/Sum_2/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
?
pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*
T0*

Tidx0*
	keep_dims( *#
_output_shapes
:?????????
R
pi/ConstConst*
_output_shapes
:*
valueB: *
dtype0
a
pi/MeanMeanpi/Sum_2pi/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
P
pi/add_10/yConst*
valueB
 *ǟ??*
_output_shapes
: *
dtype0
S
	pi/add_10Addpi/log_std/readpi/add_10/y*
T0*
_output_shapes
:
e
pi/Sum_3/reduction_indicesConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
M

pi/Const_1Const*
dtype0*
_output_shapes
: *
valueB 
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
0vf/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"<   @   *"
_class
loc:@vf/dense/kernel
?
.vf/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *??a?*
dtype0*"
_class
loc:@vf/dense/kernel
?
.vf/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@vf/dense/kernel*
valueB
 *??a>*
_output_shapes
: *
dtype0
?
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2?*
_output_shapes

:<@*
T0*

seed*"
_class
loc:@vf/dense/kernel
?
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
: 
?
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes

:<@
?
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:<@*"
_class
loc:@vf/dense/kernel
?
vf/dense/kernel
VariableV2*
	container *
dtype0*
shared_name *
_output_shapes

:<@*"
_class
loc:@vf/dense/kernel*
shape
:<@
?
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes

:<@*
validate_shape(
~
vf/dense/kernel/readIdentityvf/dense/kernel*
_output_shapes

:<@*"
_class
loc:@vf/dense/kernel*
T0
?
vf/dense/bias/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes
:@* 
_class
loc:@vf/dense/bias
?
vf/dense/bias
VariableV2*
shape:@*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
dtype0*
	container *
shared_name 
?
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
_output_shapes
:@*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
T0
t
vf/dense/bias/readIdentityvf/dense/bias*
T0*
_output_shapes
:@* 
_class
loc:@vf/dense/bias
?
vf/dense/MatMulMatMulPlaceholdervf/dense/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:?????????@*
T0
?
vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*
data_formatNHWC*'
_output_shapes
:?????????@*
T0
Y
vf/dense/TanhTanhvf/dense/BiasAdd*
T0*'
_output_shapes
:?????????@
?
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *$
_class
loc:@vf/dense_1/kernel*
_output_shapes
:*
dtype0
?
0vf/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *׳]?*$
_class
loc:@vf/dense_1/kernel*
dtype0*
_output_shapes
: 
?
0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *׳]>*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
: 
?
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*
seed2?*$
_class
loc:@vf/dense_1/kernel*

seed*
_output_shapes

:@@*
dtype0*
T0
?
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
T0
?
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
T0
?
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
T0
?
vf/dense_1/kernel
VariableV2*
shared_name *
	container *$
_class
loc:@vf/dense_1/kernel*
dtype0*
shape
:@@*
_output_shapes

:@@
?
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*
_output_shapes

:@@*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(
?
vf/dense_1/kernel/readIdentityvf/dense_1/kernel*$
_class
loc:@vf/dense_1/kernel*
_output_shapes

:@@*
T0
?
!vf/dense_1/bias/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*
dtype0*"
_class
loc:@vf/dense_1/bias
?
vf/dense_1/bias
VariableV2*
shared_name *
	container *"
_class
loc:@vf/dense_1/bias*
_output_shapes
:@*
dtype0*
shape:@
?
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*"
_class
loc:@vf/dense_1/bias
z
vf/dense_1/bias/readIdentityvf/dense_1/bias*
_output_shapes
:@*"
_class
loc:@vf/dense_1/bias*
T0
?
vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:?????????@*
transpose_b( 
?
vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*'
_output_shapes
:?????????@*
T0*
data_formatNHWC
]
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*
T0*'
_output_shapes
:?????????@
?
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*$
_class
loc:@vf/dense_2/kernel*
dtype0*
valueB"@      
?
0vf/dense_2/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *????
?
0vf/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *???>*
_output_shapes
: *
dtype0*$
_class
loc:@vf/dense_2/kernel
?
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*

seed*
T0*
_output_shapes

:@*
dtype0*
seed2?*$
_class
loc:@vf/dense_2/kernel
?
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
: 
?
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes

:@
?
,vf/dense_2/kernel/Initializer/random_uniformAdd0vf/dense_2/kernel/Initializer/random_uniform/mul0vf/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes

:@*$
_class
loc:@vf/dense_2/kernel*
T0
?
vf/dense_2/kernel
VariableV2*$
_class
loc:@vf/dense_2/kernel*
shape
:@*
_output_shapes

:@*
dtype0*
shared_name *
	container 
?
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
?
vf/dense_2/kernel/readIdentityvf/dense_2/kernel*
T0*
_output_shapes

:@*$
_class
loc:@vf/dense_2/kernel
?
!vf/dense_2/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
dtype0
?
vf/dense_2/bias
VariableV2*
shape:*
dtype0*
shared_name *"
_class
loc:@vf/dense_2/bias*
	container *
_output_shapes
:
?
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
T0*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
?
vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*'
_output_shapes
:?????????*
T0*
transpose_a( *
transpose_b( 
?
vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:?????????
?
0vc/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@vc/dense/kernel*
dtype0*
_output_shapes
:*
valueB"<   @   
?
.vc/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *??a?*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
dtype0
?
.vc/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@vc/dense/kernel*
valueB
 *??a>
?
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*

seed*
_output_shapes

:<@*
T0*"
_class
loc:@vc/dense/kernel*
dtype0*
seed2?
?
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@vc/dense/kernel
?
.vc/dense/kernel/Initializer/random_uniform/mulMul8vc/dense/kernel/Initializer/random_uniform/RandomUniform.vc/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:<@*
T0*"
_class
loc:@vc/dense/kernel
?
*vc/dense/kernel/Initializer/random_uniformAdd.vc/dense/kernel/Initializer/random_uniform/mul.vc/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
T0
?
vc/dense/kernel
VariableV2*
dtype0*
shared_name *"
_class
loc:@vc/dense/kernel*
_output_shapes

:<@*
	container *
shape
:<@
?
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
_output_shapes

:<@*
T0*"
_class
loc:@vc/dense/kernel
~
vc/dense/kernel/readIdentityvc/dense/kernel*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
T0
?
vc/dense/bias/Initializer/zerosConst*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
valueB@*    *
dtype0
?
vc/dense/bias
VariableV2* 
_class
loc:@vc/dense/bias*
shape:@*
dtype0*
_output_shapes
:@*
	container *
shared_name 
?
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
t
vc/dense/bias/readIdentityvc/dense/bias*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
:@
?
vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*'
_output_shapes
:?????????@*
transpose_b( *
transpose_a( *
T0
?
vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*'
_output_shapes
:?????????@*
data_formatNHWC*
T0
Y
vc/dense/TanhTanhvc/dense/BiasAdd*'
_output_shapes
:?????????@*
T0
?
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel
?
0vc/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel*
valueB
 *׳]?*
dtype0
?
0vc/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@vc/dense_1/kernel*
dtype0*
valueB
 *׳]>
?
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape*$
_class
loc:@vc/dense_1/kernel*
_output_shapes

:@@*
T0*

seed*
seed2?*
dtype0
?
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_1/kernel*
T0*
_output_shapes
: 
?
0vc/dense_1/kernel/Initializer/random_uniform/mulMul:vc/dense_1/kernel/Initializer/random_uniform/RandomUniform0vc/dense_1/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@vc/dense_1/kernel*
_output_shapes

:@@
?
,vc/dense_1/kernel/Initializer/random_uniformAdd0vc/dense_1/kernel/Initializer/random_uniform/mul0vc/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:@@*$
_class
loc:@vc/dense_1/kernel
?
vc/dense_1/kernel
VariableV2*
shape
:@@*$
_class
loc:@vc/dense_1/kernel*
shared_name *
_output_shapes

:@@*
	container *
dtype0
?
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@*
use_locking(
?
vc/dense_1/kernel/readIdentityvc/dense_1/kernel*
_output_shapes

:@@*
T0*$
_class
loc:@vc/dense_1/kernel
?
!vc/dense_1/bias/Initializer/zerosConst*"
_class
loc:@vc/dense_1/bias*
dtype0*
valueB@*    *
_output_shapes
:@
?
vc/dense_1/bias
VariableV2*
shared_name *
dtype0*"
_class
loc:@vc/dense_1/bias*
	container *
_output_shapes
:@*
shape:@
?
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*
_output_shapes
:@*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0
z
vc/dense_1/bias/readIdentityvc/dense_1/bias*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes
:@
?
vc/dense_1/MatMulMatMulvc/dense/Tanhvc/dense_1/kernel/read*
transpose_a( *'
_output_shapes
:?????????@*
transpose_b( *
T0
?
vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*
T0*'
_output_shapes
:?????????@*
data_formatNHWC
]
vc/dense_1/TanhTanhvc/dense_1/BiasAdd*'
_output_shapes
:?????????@*
T0
?
2vc/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:*
dtype0
?
0vc/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *$
_class
loc:@vc/dense_2/kernel*
valueB
 *????
?
0vc/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *???>*$
_class
loc:@vc/dense_2/kernel
?
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*
seed2?*
dtype0*
T0*

seed*
_output_shapes

:@*$
_class
loc:@vc/dense_2/kernel
?
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *$
_class
loc:@vc/dense_2/kernel
?
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes

:@
?
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes

:@
?
vc/dense_2/kernel
VariableV2*
dtype0*
	container *$
_class
loc:@vc/dense_2/kernel*
shared_name *
shape
:@*
_output_shapes

:@
?
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@vc/dense_2/kernel*
use_locking(
?
vc/dense_2/kernel/readIdentityvc/dense_2/kernel*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes

:@
?
!vc/dense_2/bias/Initializer/zerosConst*"
_class
loc:@vc/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
?
vc/dense_2/bias
VariableV2*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
shape:*
shared_name *
	container *
dtype0
?
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias
?
vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*
transpose_a( *'
_output_shapes
:?????????*
T0*
transpose_b( 
?
vc/dense_2/BiasAddBiasAddvc/dense_2/MatMulvc/dense_2/bias/read*'
_output_shapes
:?????????*
T0*
data_formatNHWC
n

vc/SqueezeSqueezevc/dense_2/BiasAdd*
T0*#
_output_shapes
:?????????*
squeeze_dims

?
1lam/dense/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@lam/dense/kernel*
dtype0*
valueB"<   @   *
_output_shapes
:
?
/lam/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *??a?*
dtype0*
_output_shapes
: *#
_class
loc:@lam/dense/kernel
?
/lam/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *#
_class
loc:@lam/dense/kernel*
valueB
 *??a>*
dtype0
?
9lam/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform1lam/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:<@*
seed2?*
T0*#
_class
loc:@lam/dense/kernel*

seed
?
/lam/dense/kernel/Initializer/random_uniform/subSub/lam/dense/kernel/Initializer/random_uniform/max/lam/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *#
_class
loc:@lam/dense/kernel*
T0
?
/lam/dense/kernel/Initializer/random_uniform/mulMul9lam/dense/kernel/Initializer/random_uniform/RandomUniform/lam/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:<@*
T0*#
_class
loc:@lam/dense/kernel
?
+lam/dense/kernel/Initializer/random_uniformAdd/lam/dense/kernel/Initializer/random_uniform/mul/lam/dense/kernel/Initializer/random_uniform/min*#
_class
loc:@lam/dense/kernel*
T0*
_output_shapes

:<@
?
lam/dense/kernel
VariableV2*
shared_name *
dtype0*
	container *#
_class
loc:@lam/dense/kernel*
shape
:<@*
_output_shapes

:<@
?
lam/dense/kernel/AssignAssignlam/dense/kernel+lam/dense/kernel/Initializer/random_uniform*
T0*
validate_shape(*#
_class
loc:@lam/dense/kernel*
_output_shapes

:<@*
use_locking(
?
lam/dense/kernel/readIdentitylam/dense/kernel*
_output_shapes

:<@*
T0*#
_class
loc:@lam/dense/kernel
?
 lam/dense/bias/Initializer/zerosConst*!
_class
loc:@lam/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
?
lam/dense/bias
VariableV2*
_output_shapes
:@*!
_class
loc:@lam/dense/bias*
dtype0*
shape:@*
shared_name *
	container 
?
lam/dense/bias/AssignAssignlam/dense/bias lam/dense/bias/Initializer/zeros*!
_class
loc:@lam/dense/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
w
lam/dense/bias/readIdentitylam/dense/bias*!
_class
loc:@lam/dense/bias*
_output_shapes
:@*
T0
?
lam/dense/MatMulMatMulPlaceholderlam/dense/kernel/read*'
_output_shapes
:?????????@*
transpose_a( *
transpose_b( *
T0
?
lam/dense/BiasAddBiasAddlam/dense/MatMullam/dense/bias/read*'
_output_shapes
:?????????@*
T0*
data_formatNHWC
[
lam/dense/TanhTanhlam/dense/BiasAdd*'
_output_shapes
:?????????@*
T0
?
3lam/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*%
_class
loc:@lam/dense_1/kernel*
_output_shapes
:*
valueB"@   @   
?
1lam/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*%
_class
loc:@lam/dense_1/kernel*
valueB
 *׳]?*
_output_shapes
: 
?
1lam/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*%
_class
loc:@lam/dense_1/kernel*
_output_shapes
: *
dtype0
?
;lam/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform3lam/dense_1/kernel/Initializer/random_uniform/shape*%
_class
loc:@lam/dense_1/kernel*
seed2?*
_output_shapes

:@@*
T0*
dtype0*

seed
?
1lam/dense_1/kernel/Initializer/random_uniform/subSub1lam/dense_1/kernel/Initializer/random_uniform/max1lam/dense_1/kernel/Initializer/random_uniform/min*%
_class
loc:@lam/dense_1/kernel*
_output_shapes
: *
T0
?
1lam/dense_1/kernel/Initializer/random_uniform/mulMul;lam/dense_1/kernel/Initializer/random_uniform/RandomUniform1lam/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*%
_class
loc:@lam/dense_1/kernel
?
-lam/dense_1/kernel/Initializer/random_uniformAdd1lam/dense_1/kernel/Initializer/random_uniform/mul1lam/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel*
T0
?
lam/dense_1/kernel
VariableV2*
shape
:@@*
dtype0*
	container *
shared_name *
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel
?
lam/dense_1/kernel/AssignAssignlam/dense_1/kernel-lam/dense_1/kernel/Initializer/random_uniform*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel*
validate_shape(*
T0*
use_locking(
?
lam/dense_1/kernel/readIdentitylam/dense_1/kernel*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel*
T0
?
"lam/dense_1/bias/Initializer/zerosConst*
dtype0*#
_class
loc:@lam/dense_1/bias*
_output_shapes
:@*
valueB@*    
?
lam/dense_1/bias
VariableV2*
shared_name *#
_class
loc:@lam/dense_1/bias*
dtype0*
_output_shapes
:@*
	container *
shape:@
?
lam/dense_1/bias/AssignAssignlam/dense_1/bias"lam/dense_1/bias/Initializer/zeros*
_output_shapes
:@*
T0*
validate_shape(*#
_class
loc:@lam/dense_1/bias*
use_locking(
}
lam/dense_1/bias/readIdentitylam/dense_1/bias*#
_class
loc:@lam/dense_1/bias*
_output_shapes
:@*
T0
?
lam/dense_1/MatMulMatMullam/dense/Tanhlam/dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:?????????@
?
lam/dense_1/BiasAddBiasAddlam/dense_1/MatMullam/dense_1/bias/read*
T0*'
_output_shapes
:?????????@*
data_formatNHWC
_
lam/dense_1/TanhTanhlam/dense_1/BiasAdd*'
_output_shapes
:?????????@*
T0
?
3lam/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
_class
loc:@lam/dense_2/kernel*
valueB"@      
?
1lam/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *????*
_output_shapes
: *
dtype0*%
_class
loc:@lam/dense_2/kernel
?
1lam/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *???>*%
_class
loc:@lam/dense_2/kernel*
dtype0
?
;lam/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform3lam/dense_2/kernel/Initializer/random_uniform/shape*
seed2?*

seed*
dtype0*
T0*%
_class
loc:@lam/dense_2/kernel*
_output_shapes

:@
?
1lam/dense_2/kernel/Initializer/random_uniform/subSub1lam/dense_2/kernel/Initializer/random_uniform/max1lam/dense_2/kernel/Initializer/random_uniform/min*%
_class
loc:@lam/dense_2/kernel*
_output_shapes
: *
T0
?
1lam/dense_2/kernel/Initializer/random_uniform/mulMul;lam/dense_2/kernel/Initializer/random_uniform/RandomUniform1lam/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel
?
-lam/dense_2/kernel/Initializer/random_uniformAdd1lam/dense_2/kernel/Initializer/random_uniform/mul1lam/dense_2/kernel/Initializer/random_uniform/min*
T0*%
_class
loc:@lam/dense_2/kernel*
_output_shapes

:@
?
lam/dense_2/kernel
VariableV2*
shape
:@*
_output_shapes

:@*
shared_name *
	container *%
_class
loc:@lam/dense_2/kernel*
dtype0
?
lam/dense_2/kernel/AssignAssignlam/dense_2/kernel-lam/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel
?
lam/dense_2/kernel/readIdentitylam/dense_2/kernel*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel*
T0
?
"lam/dense_2/bias/Initializer/ConstConst*
_output_shapes
:*
valueB*  ??*#
_class
loc:@lam/dense_2/bias*
dtype0
?
lam/dense_2/bias
VariableV2*#
_class
loc:@lam/dense_2/bias*
	container *
dtype0*
_output_shapes
:*
shape:*
shared_name 
?
lam/dense_2/bias/AssignAssignlam/dense_2/bias"lam/dense_2/bias/Initializer/Const*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*#
_class
loc:@lam/dense_2/bias
}
lam/dense_2/bias/readIdentitylam/dense_2/bias*
_output_shapes
:*#
_class
loc:@lam/dense_2/bias*
T0
?
lam/dense_2/MatMulMatMullam/dense_1/Tanhlam/dense_2/kernel/read*'
_output_shapes
:?????????*
T0*
transpose_a( *
transpose_b( 
?
lam/dense_2/BiasAddBiasAddlam/dense_2/MatMullam/dense_2/bias/read*'
_output_shapes
:?????????*
data_formatNHWC*
T0
g
lam/dense_2/SoftplusSoftpluslam/dense_2/BiasAdd*'
_output_shapes
:?????????*
T0
q
lam/SqueezeSqueezelam/dense_2/Softplus*#
_output_shapes
:?????????*
squeeze_dims
*
T0
@
NegNegpi/Sum*#
_output_shapes
:?????????*
T0
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
V
MeanMeanNegConst*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
O
subSubpi/SumPlaceholder_6*#
_output_shapes
:?????????*
T0
=
ExpExpsub*
T0*#
_output_shapes
:?????????
N
	Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
GreaterGreaterPlaceholder_2	Greater/y*#
_output_shapes
:?????????*
T0
J
mul/xConst*
_output_shapes
: *
valueB
 *????*
dtype0
N
mulMulmul/xPlaceholder_2*#
_output_shapes
:?????????*
T0
L
mul_1/xConst*
valueB
 *??L?*
dtype0*
_output_shapes
: 
R
mul_1Mulmul_1/xPlaceholder_2*#
_output_shapes
:?????????*
T0
S
SelectSelectGreatermulmul_1*#
_output_shapes
:?????????*
T0
N
mul_2MulExpPlaceholder_2*
T0*#
_output_shapes
:?????????
O
MinimumMinimummul_2Select*
T0*#
_output_shapes
:?????????
Q
Const_1Const*
_output_shapes
:*
valueB: *
dtype0
^
Mean_1MeanMinimumConst_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
N
mul_3MulExpPlaceholder_3*
T0*#
_output_shapes
:?????????
W
StopGradientStopGradientlam/Squeeze*
T0*#
_output_shapes
:?????????
O
Mul_4MulStopGradientmul_3*#
_output_shapes
:?????????*
T0
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2MeanMul_4Const_2*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
\
clip_by_value/Minimum/yConst*
_output_shapes
: *
valueB
 *  ?B*
dtype0
v
clip_by_value/MinimumMinimumPlaceholder_7clip_by_value/Minimum/y*
T0*#
_output_shapes
:?????????
T
clip_by_value/yConst*
dtype0*
valueB
 *??L?*
_output_shapes
: 
n
clip_by_valueMaximumclip_by_value/Minimumclip_by_value/y*#
_output_shapes
:?????????*
T0
[
StopGradient_1StopGradientclip_by_value*#
_output_shapes
:?????????*
T0
W
Mul_5Mullam/SqueezeStopGradient_1*
T0*#
_output_shapes
:?????????
Q
Const_3Const*
dtype0*
valueB: *
_output_shapes
:
\
Mean_3MeanMul_5Const_3*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
L
mul_6/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
A
mul_6Mulmul_6/x	pi/Mean_1*
_output_shapes
: *
T0
:
addAddMean_1mul_6*
_output_shapes
: *
T0
:
sub_1SubaddMean_2*
T0*
_output_shapes
: 
4
Neg_1Negsub_1*
T0*
_output_shapes
: 
5
Neg_2NegMean_3*
_output_shapes
: *
T0
[
Variable/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
shape: *
	container *
_output_shapes
: *
dtype0*
shared_name 
?
Variable/AssignAssignVariableVariable/initial_value*
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable*
_output_shapes
: 
a
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes
: 
b
PolynomialDecay/learning_rateConst*
valueB
 *RI?9*
dtype0*
_output_shapes
: 
[
PolynomialDecay/Cast/xConst*
valueB
 *?7?5*
dtype0*
_output_shapes
: 
]
PolynomialDecay/Cast_1/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
[
PolynomialDecay/Cast_3/xConst*
value
B :?]*
dtype0*
_output_shapes
: 
x
PolynomialDecay/Cast_3CastPolynomialDecay/Cast_3/x*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
^
PolynomialDecay/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?;F
m
PolynomialDecay/MinimumMinimumVariable/readPolynomialDecay/Minimum/y*
_output_shapes
: *
T0
p
PolynomialDecay/divRealDivPolynomialDecay/MinimumPolynomialDecay/Cast_3*
T0*
_output_shapes
: 
r
PolynomialDecay/subSubPolynomialDecay/learning_ratePolynomialDecay/Cast/x*
_output_shapes
: *
T0
\
PolynomialDecay/sub_1/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
k
PolynomialDecay/sub_1SubPolynomialDecay/sub_1/xPolynomialDecay/div*
T0*
_output_shapes
: 
l
PolynomialDecay/PowPowPolynomialDecay/sub_1PolynomialDecay/Cast_1/x*
_output_shapes
: *
T0
e
PolynomialDecay/MulMulPolynomialDecay/subPolynomialDecay/Pow*
_output_shapes
: *
T0
d
PolynomialDecayAddPolynomialDecay/MulPolynomialDecay/Cast/x*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  ??*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
P
gradients/Neg_1_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
Z
gradients/sub_1_grad/NegNeggradients/Neg_1_grad/Neg*
T0*
_output_shapes
: 
c
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/Neg_1_grad/Neg^gradients/sub_1_grad/Neg
?
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/Neg_1_grad/Neg&^gradients/sub_1_grad/tuple/group_deps*
_output_shapes
: *+
_class!
loc:@gradients/Neg_1_grad/Neg*
T0
?
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Neg&^gradients/sub_1_grad/tuple/group_deps*+
_class!
loc:@gradients/sub_1_grad/Neg*
_output_shapes
: *
T0
[
#gradients/add_grad/tuple/group_depsNoOp.^gradients/sub_1_grad/tuple/control_dependency
?
+gradients/add_grad/tuple/control_dependencyIdentity-gradients/sub_1_grad/tuple/control_dependency$^gradients/add_grad/tuple/group_deps*+
_class!
loc:@gradients/Neg_1_grad/Neg*
_output_shapes
: *
T0
?
-gradients/add_grad/tuple/control_dependency_1Identity-gradients/sub_1_grad/tuple/control_dependency$^gradients/add_grad/tuple/group_deps*+
_class!
loc:@gradients/Neg_1_grad/Neg*
_output_shapes
: *
T0
m
#gradients/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
gradients/Mean_2_grad/ReshapeReshape/gradients/sub_1_grad/tuple/control_dependency_1#gradients/Mean_2_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
`
gradients/Mean_2_grad/ShapeShapeMul_4*
_output_shapes
:*
out_type0*
T0
?
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*#
_output_shapes
:?????????*
T0*

Tmultiples0
b
gradients/Mean_2_grad/Shape_1ShapeMul_4*
out_type0*
T0*
_output_shapes
:
`
gradients/Mean_2_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
e
gradients/Mean_2_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
g
gradients/Mean_2_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
?
gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
a
gradients/Mean_2_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
?
gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
?
gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
_output_shapes
: *
T0
?
gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
?
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
T0*#
_output_shapes
:?????????
m
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
?
gradients/Mean_1_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency#gradients/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients/Mean_1_grad/ShapeShapeMinimum*
_output_shapes
:*
out_type0*
T0
?
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:?????????
d
gradients/Mean_1_grad/Shape_1ShapeMinimum*
out_type0*
_output_shapes
:*
T0
`
gradients/Mean_1_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
e
gradients/Mean_1_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
?
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
g
gradients/Mean_1_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
?
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
?
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
?
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*#
_output_shapes
:?????????
z
gradients/mul_6_grad/MulMul-gradients/add_grad/tuple/control_dependency_1	pi/Mean_1*
T0*
_output_shapes
: 
z
gradients/mul_6_grad/Mul_1Mul-gradients/add_grad/tuple/control_dependency_1mul_6/x*
_output_shapes
: *
T0
e
%gradients/mul_6_grad/tuple/group_depsNoOp^gradients/mul_6_grad/Mul^gradients/mul_6_grad/Mul_1
?
-gradients/mul_6_grad/tuple/control_dependencyIdentitygradients/mul_6_grad/Mul&^gradients/mul_6_grad/tuple/group_deps*+
_class!
loc:@gradients/mul_6_grad/Mul*
_output_shapes
: *
T0
?
/gradients/mul_6_grad/tuple/control_dependency_1Identitygradients/mul_6_grad/Mul_1&^gradients/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_6_grad/Mul_1*
_output_shapes
: 
f
gradients/Mul_4_grad/ShapeShapeStopGradient*
_output_shapes
:*
out_type0*
T0
a
gradients/Mul_4_grad/Shape_1Shapemul_3*
out_type0*
_output_shapes
:*
T0
?
*gradients/Mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_4_grad/Shapegradients/Mul_4_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
s
gradients/Mul_4_grad/MulMulgradients/Mean_2_grad/truedivmul_3*
T0*#
_output_shapes
:?????????
?
gradients/Mul_4_grad/SumSumgradients/Mul_4_grad/Mul*gradients/Mul_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
gradients/Mul_4_grad/ReshapeReshapegradients/Mul_4_grad/Sumgradients/Mul_4_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
|
gradients/Mul_4_grad/Mul_1MulStopGradientgradients/Mean_2_grad/truediv*#
_output_shapes
:?????????*
T0
?
gradients/Mul_4_grad/Sum_1Sumgradients/Mul_4_grad/Mul_1,gradients/Mul_4_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients/Mul_4_grad/Reshape_1Reshapegradients/Mul_4_grad/Sum_1gradients/Mul_4_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
m
%gradients/Mul_4_grad/tuple/group_depsNoOp^gradients/Mul_4_grad/Reshape^gradients/Mul_4_grad/Reshape_1
?
-gradients/Mul_4_grad/tuple/control_dependencyIdentitygradients/Mul_4_grad/Reshape&^gradients/Mul_4_grad/tuple/group_deps*#
_output_shapes
:?????????*/
_class%
#!loc:@gradients/Mul_4_grad/Reshape*
T0
?
/gradients/Mul_4_grad/tuple/control_dependency_1Identitygradients/Mul_4_grad/Reshape_1&^gradients/Mul_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Mul_4_grad/Reshape_1*
T0*#
_output_shapes
:?????????
a
gradients/Minimum_grad/ShapeShapemul_2*
T0*
_output_shapes
:*
out_type0
d
gradients/Minimum_grad/Shape_1ShapeSelect*
_output_shapes
:*
T0*
out_type0
{
gradients/Minimum_grad/Shape_2Shapegradients/Mean_1_grad/truediv*
T0*
out_type0*
_output_shapes
:
g
"gradients/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*#
_output_shapes
:?????????*

index_type0*
T0
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*#
_output_shapes
:?????????*
T0
?
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_1_grad/truedivgradients/Minimum_grad/zeros*#
_output_shapes
:?????????*
T0
?
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_1_grad/truediv*#
_output_shapes
:?????????*
T0
?
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*#
_output_shapes
:?????????*
Tshape0
?
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
Tshape0*#
_output_shapes
:?????????*
T0
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
?
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Minimum_grad/Reshape*
T0*#
_output_shapes
:?????????
?
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*#
_output_shapes
:?????????
i
&gradients/pi/Mean_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
 gradients/pi/Mean_1_grad/ReshapeReshape/gradients/mul_6_grad/tuple/control_dependency_1&gradients/pi/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
: *
Tshape0
a
gradients/pi/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB 
?
gradients/pi/Mean_1_grad/TileTile gradients/pi/Mean_1_grad/Reshapegradients/pi/Mean_1_grad/Const*
T0*

Tmultiples0*
_output_shapes
: 
e
 gradients/pi/Mean_1_grad/Const_1Const*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
 gradients/pi/Mean_1_grad/truedivRealDivgradients/pi/Mean_1_grad/Tile gradients/pi/Mean_1_grad/Const_1*
_output_shapes
: *
T0
]
gradients/mul_3_grad/ShapeShapeExp*
T0*
_output_shapes
:*
out_type0
i
gradients/mul_3_grad/Shape_1ShapePlaceholder_3*
_output_shapes
:*
out_type0*
T0
?
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/mul_3_grad/MulMul/gradients/Mul_4_grad/tuple/control_dependency_1Placeholder_3*#
_output_shapes
:?????????*
T0
?
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
?
gradients/mul_3_grad/Mul_1MulExp/gradients/Mul_4_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:?????????
?
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
Tshape0*#
_output_shapes
:?????????*
T0
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
?
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*#
_output_shapes
:?????????
?
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*#
_output_shapes
:?????????
]
gradients/mul_2_grad/ShapeShapeExp*
out_type0*
T0*
_output_shapes
:
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_2*
out_type0*
T0*
_output_shapes
:
?
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_2*
T0*#
_output_shapes
:?????????
?
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*#
_output_shapes
:?????????*
T0
?
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*#
_output_shapes
:?????????*
T0*
Tshape0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
?
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
?
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1
g
gradients/pi/Sum_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
?
gradients/pi/Sum_3_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
gradients/pi/Sum_3_grad/addAddpi/Sum_3/reduction_indicesgradients/pi/Sum_3_grad/Size*
T0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: 
?
gradients/pi/Sum_3_grad/modFloorModgradients/pi/Sum_3_grad/addgradients/pi/Sum_3_grad/Size*
_output_shapes
: *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
T0
?
gradients/pi/Sum_3_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
#gradients/pi/Sum_3_grad/range/startConst*
value	B : *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: *
dtype0
?
#gradients/pi/Sum_3_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
gradients/pi/Sum_3_grad/rangeRange#gradients/pi/Sum_3_grad/range/startgradients/pi/Sum_3_grad/Size#gradients/pi/Sum_3_grad/range/delta*
_output_shapes
:*

Tidx0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
"gradients/pi/Sum_3_grad/Fill/valueConst*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
value	B :*
_output_shapes
: *
dtype0
?
gradients/pi/Sum_3_grad/FillFillgradients/pi/Sum_3_grad/Shape_1"gradients/pi/Sum_3_grad/Fill/value*

index_type0*
_output_shapes
: *
T0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
%gradients/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients/pi/Sum_3_grad/rangegradients/pi/Sum_3_grad/modgradients/pi/Sum_3_grad/Shapegradients/pi/Sum_3_grad/Fill*
T0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
N*
_output_shapes
:
?
!gradients/pi/Sum_3_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
gradients/pi/Sum_3_grad/MaximumMaximum%gradients/pi/Sum_3_grad/DynamicStitch!gradients/pi/Sum_3_grad/Maximum/y*
_output_shapes
:*
T0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
 gradients/pi/Sum_3_grad/floordivFloorDivgradients/pi/Sum_3_grad/Shapegradients/pi/Sum_3_grad/Maximum*
_output_shapes
:*
T0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
gradients/pi/Sum_3_grad/ReshapeReshape gradients/pi/Mean_1_grad/truediv%gradients/pi/Sum_3_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
?
gradients/pi/Sum_3_grad/TileTilegradients/pi/Sum_3_grad/Reshape gradients/pi/Sum_3_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
?
gradients/AddNAddN-gradients/mul_3_grad/tuple/control_dependency-gradients/mul_2_grad/tuple/control_dependency*
N*#
_output_shapes
:?????????*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape
`
gradients/Exp_grad/mulMulgradients/AddNExp*#
_output_shapes
:?????????*
T0
h
gradients/pi/add_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
c
 gradients/pi/add_10_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
.gradients/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_10_grad/Shape gradients/pi/add_10_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/pi/add_10_grad/SumSumgradients/pi/Sum_3_grad/Tile.gradients/pi/add_10_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
 gradients/pi/add_10_grad/ReshapeReshapegradients/pi/add_10_grad/Sumgradients/pi/add_10_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
gradients/pi/add_10_grad/Sum_1Sumgradients/pi/Sum_3_grad/Tile0gradients/pi/add_10_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
"gradients/pi/add_10_grad/Reshape_1Reshapegradients/pi/add_10_grad/Sum_1 gradients/pi/add_10_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
y
)gradients/pi/add_10_grad/tuple/group_depsNoOp!^gradients/pi/add_10_grad/Reshape#^gradients/pi/add_10_grad/Reshape_1
?
1gradients/pi/add_10_grad/tuple/control_dependencyIdentity gradients/pi/add_10_grad/Reshape*^gradients/pi/add_10_grad/tuple/group_deps*
_output_shapes
:*
T0*3
_class)
'%loc:@gradients/pi/add_10_grad/Reshape
?
3gradients/pi/add_10_grad/tuple/control_dependency_1Identity"gradients/pi/add_10_grad/Reshape_1*^gradients/pi/add_10_grad/tuple/group_deps*
T0*
_output_shapes
: *5
_class+
)'loc:@gradients/pi/add_10_grad/Reshape_1
^
gradients/sub_grad/ShapeShapepi/Sum*
out_type0*
_output_shapes
:*
T0
g
gradients/sub_grad/Shape_1ShapePlaceholder_6*
out_type0*
T0*
_output_shapes
:
?
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
?
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
?
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*#
_output_shapes
:?????????*
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
?
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:?????????*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0
?
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:?????????*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0
c
gradients/pi/Sum_grad/ShapeShapepi/mul_2*
T0*
out_type0*
_output_shapes
:
?
gradients/pi/Sum_grad/SizeConst*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0
?
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
?
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*
_output_shapes
: 
?
gradients/pi/Sum_grad/Shape_1Const*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
valueB *
_output_shapes
: 
?
!gradients/pi/Sum_grad/range/startConst*
value	B : *
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
?
!gradients/pi/Sum_grad/range/deltaConst*
dtype0*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
?
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*

Tidx0*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
?
 gradients/pi/Sum_grad/Fill/valueConst*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
value	B :
?
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
T0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*

index_type0
?
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
N*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*
_output_shapes
:
?
gradients/pi/Sum_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
?
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0
?
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*
T0*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
?
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*0
_output_shapes
:??????????????????*
T0*
Tshape0
?
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:?????????
`
gradients/pi/mul_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
g
gradients/pi/mul_2_grad/Shape_1Shapepi/add_3*
out_type0*
_output_shapes
:*
T0
?
-gradients/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_2_grad/Shapegradients/pi/mul_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
z
gradients/pi/mul_2_grad/MulMulgradients/pi/Sum_grad/Tilepi/add_3*'
_output_shapes
:?????????*
T0
?
gradients/pi/mul_2_grad/SumSumgradients/pi/mul_2_grad/Mul-gradients/pi/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
gradients/pi/mul_2_grad/ReshapeReshapegradients/pi/mul_2_grad/Sumgradients/pi/mul_2_grad/Shape*
T0*
_output_shapes
: *
Tshape0
~
gradients/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients/pi/Sum_grad/Tile*'
_output_shapes
:?????????*
T0
?
gradients/pi/mul_2_grad/Sum_1Sumgradients/pi/mul_2_grad/Mul_1/gradients/pi/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
!gradients/pi/mul_2_grad/Reshape_1Reshapegradients/pi/mul_2_grad/Sum_1gradients/pi/mul_2_grad/Shape_1*'
_output_shapes
:?????????*
Tshape0*
T0
v
(gradients/pi/mul_2_grad/tuple/group_depsNoOp ^gradients/pi/mul_2_grad/Reshape"^gradients/pi/mul_2_grad/Reshape_1
?
0gradients/pi/mul_2_grad/tuple/control_dependencyIdentitygradients/pi/mul_2_grad/Reshape)^gradients/pi/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
: *2
_class(
&$loc:@gradients/pi/mul_2_grad/Reshape
?
2gradients/pi/mul_2_grad/tuple/control_dependency_1Identity!gradients/pi/mul_2_grad/Reshape_1)^gradients/pi/mul_2_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*4
_class*
(&loc:@gradients/pi/mul_2_grad/Reshape_1
e
gradients/pi/add_3_grad/ShapeShapepi/add_2*
T0*
_output_shapes
:*
out_type0
b
gradients/pi/add_3_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
?
-gradients/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_3_grad/Shapegradients/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/pi/add_3_grad/SumSum2gradients/pi/mul_2_grad/tuple/control_dependency_1-gradients/pi/add_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
gradients/pi/add_3_grad/ReshapeReshapegradients/pi/add_3_grad/Sumgradients/pi/add_3_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients/pi/add_3_grad/Sum_1Sum2gradients/pi/mul_2_grad/tuple/control_dependency_1/gradients/pi/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
?
!gradients/pi/add_3_grad/Reshape_1Reshapegradients/pi/add_3_grad/Sum_1gradients/pi/add_3_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
v
(gradients/pi/add_3_grad/tuple/group_depsNoOp ^gradients/pi/add_3_grad/Reshape"^gradients/pi/add_3_grad/Reshape_1
?
0gradients/pi/add_3_grad/tuple/control_dependencyIdentitygradients/pi/add_3_grad/Reshape)^gradients/pi/add_3_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*2
_class(
&$loc:@gradients/pi/add_3_grad/Reshape
?
2gradients/pi/add_3_grad/tuple/control_dependency_1Identity!gradients/pi/add_3_grad/Reshape_1)^gradients/pi/add_3_grad/tuple/group_deps*
_output_shapes
: *4
_class*
(&loc:@gradients/pi/add_3_grad/Reshape_1*
T0
c
gradients/pi/add_2_grad/ShapeShapepi/pow*
_output_shapes
:*
T0*
out_type0
i
gradients/pi/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
?
-gradients/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_2_grad/Shapegradients/pi/add_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/pi/add_2_grad/SumSum0gradients/pi/add_3_grad/tuple/control_dependency-gradients/pi/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/pi/add_2_grad/ReshapeReshapegradients/pi/add_2_grad/Sumgradients/pi/add_2_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
gradients/pi/add_2_grad/Sum_1Sum0gradients/pi/add_3_grad/tuple/control_dependency/gradients/pi/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
?
!gradients/pi/add_2_grad/Reshape_1Reshapegradients/pi/add_2_grad/Sum_1gradients/pi/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
v
(gradients/pi/add_2_grad/tuple/group_depsNoOp ^gradients/pi/add_2_grad/Reshape"^gradients/pi/add_2_grad/Reshape_1
?
0gradients/pi/add_2_grad/tuple/control_dependencyIdentitygradients/pi/add_2_grad/Reshape)^gradients/pi/add_2_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*2
_class(
&$loc:@gradients/pi/add_2_grad/Reshape
?
2gradients/pi/add_2_grad/tuple/control_dependency_1Identity!gradients/pi/add_2_grad/Reshape_1)^gradients/pi/add_2_grad/tuple/group_deps*
_output_shapes
:*
T0*4
_class*
(&loc:@gradients/pi/add_2_grad/Reshape_1
e
gradients/pi/pow_grad/ShapeShape
pi/truediv*
_output_shapes
:*
out_type0*
T0
`
gradients/pi/pow_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
?
+gradients/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/pow_grad/Shapegradients/pi/pow_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/pi/pow_grad/mulMul0gradients/pi/add_2_grad/tuple/control_dependencypi/pow/y*
T0*'
_output_shapes
:?????????
`
gradients/pi/pow_grad/sub/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
h
gradients/pi/pow_grad/subSubpi/pow/ygradients/pi/pow_grad/sub/y*
_output_shapes
: *
T0
y
gradients/pi/pow_grad/PowPow
pi/truedivgradients/pi/pow_grad/sub*'
_output_shapes
:?????????*
T0
?
gradients/pi/pow_grad/mul_1Mulgradients/pi/pow_grad/mulgradients/pi/pow_grad/Pow*
T0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/SumSumgradients/pi/pow_grad/mul_1+gradients/pi/pow_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients/pi/pow_grad/ReshapeReshapegradients/pi/pow_grad/Sumgradients/pi/pow_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
d
gradients/pi/pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
gradients/pi/pow_grad/GreaterGreater
pi/truedivgradients/pi/pow_grad/Greater/y*'
_output_shapes
:?????????*
T0
o
%gradients/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
j
%gradients/pi/pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
gradients/pi/pow_grad/ones_likeFill%gradients/pi/pow_grad/ones_like/Shape%gradients/pi/pow_grad/ones_like/Const*'
_output_shapes
:?????????*
T0*

index_type0
?
gradients/pi/pow_grad/SelectSelectgradients/pi/pow_grad/Greater
pi/truedivgradients/pi/pow_grad/ones_like*'
_output_shapes
:?????????*
T0
p
gradients/pi/pow_grad/LogLoggradients/pi/pow_grad/Select*
T0*'
_output_shapes
:?????????
k
 gradients/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*
T0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/Select_1Selectgradients/pi/pow_grad/Greatergradients/pi/pow_grad/Log gradients/pi/pow_grad/zeros_like*
T0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/mul_2Mul0gradients/pi/add_2_grad/tuple/control_dependencypi/pow*
T0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/mul_3Mulgradients/pi/pow_grad/mul_2gradients/pi/pow_grad/Select_1*'
_output_shapes
:?????????*
T0
?
gradients/pi/pow_grad/Sum_1Sumgradients/pi/pow_grad/mul_3-gradients/pi/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
gradients/pi/pow_grad/Reshape_1Reshapegradients/pi/pow_grad/Sum_1gradients/pi/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/pi/pow_grad/tuple/group_depsNoOp^gradients/pi/pow_grad/Reshape ^gradients/pi/pow_grad/Reshape_1
?
.gradients/pi/pow_grad/tuple/control_dependencyIdentitygradients/pi/pow_grad/Reshape'^gradients/pi/pow_grad/tuple/group_deps*'
_output_shapes
:?????????*0
_class&
$"loc:@gradients/pi/pow_grad/Reshape*
T0
?
0gradients/pi/pow_grad/tuple/control_dependency_1Identitygradients/pi/pow_grad/Reshape_1'^gradients/pi/pow_grad/tuple/group_deps*
T0*
_output_shapes
: *2
_class(
&$loc:@gradients/pi/pow_grad/Reshape_1
`
gradients/pi/mul_1_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
i
gradients/pi/mul_1_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
?
-gradients/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_1_grad/Shapegradients/pi/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/pi/mul_1_grad/MulMul2gradients/pi/add_2_grad/tuple/control_dependency_1pi/log_std/read*
T0*
_output_shapes
:
?
gradients/pi/mul_1_grad/SumSumgradients/pi/mul_1_grad/Mul-gradients/pi/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
?
gradients/pi/mul_1_grad/ReshapeReshapegradients/pi/mul_1_grad/Sumgradients/pi/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
?
gradients/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x2gradients/pi/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
?
gradients/pi/mul_1_grad/Sum_1Sumgradients/pi/mul_1_grad/Mul_1/gradients/pi/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
!gradients/pi/mul_1_grad/Reshape_1Reshapegradients/pi/mul_1_grad/Sum_1gradients/pi/mul_1_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0
v
(gradients/pi/mul_1_grad/tuple/group_depsNoOp ^gradients/pi/mul_1_grad/Reshape"^gradients/pi/mul_1_grad/Reshape_1
?
0gradients/pi/mul_1_grad/tuple/control_dependencyIdentitygradients/pi/mul_1_grad/Reshape)^gradients/pi/mul_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/mul_1_grad/Reshape*
_output_shapes
: 
?
2gradients/pi/mul_1_grad/tuple/control_dependency_1Identity!gradients/pi/mul_1_grad/Reshape_1)^gradients/pi/mul_1_grad/tuple/group_deps*
_output_shapes
:*
T0*4
_class*
(&loc:@gradients/pi/mul_1_grad/Reshape_1
e
gradients/pi/truediv_grad/ShapeShapepi/sub*
_output_shapes
:*
T0*
out_type0
k
!gradients/pi/truediv_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
?
/gradients/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/truediv_grad/Shape!gradients/pi/truediv_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
!gradients/pi/truediv_grad/RealDivRealDiv.gradients/pi/pow_grad/tuple/control_dependencypi/add_1*
T0*'
_output_shapes
:?????????
?
gradients/pi/truediv_grad/SumSum!gradients/pi/truediv_grad/RealDiv/gradients/pi/truediv_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
?
!gradients/pi/truediv_grad/ReshapeReshapegradients/pi/truediv_grad/Sumgradients/pi/truediv_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
^
gradients/pi/truediv_grad/NegNegpi/sub*'
_output_shapes
:?????????*
T0
?
#gradients/pi/truediv_grad/RealDiv_1RealDivgradients/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:?????????*
T0
?
#gradients/pi/truediv_grad/RealDiv_2RealDiv#gradients/pi/truediv_grad/RealDiv_1pi/add_1*'
_output_shapes
:?????????*
T0
?
gradients/pi/truediv_grad/mulMul.gradients/pi/pow_grad/tuple/control_dependency#gradients/pi/truediv_grad/RealDiv_2*'
_output_shapes
:?????????*
T0
?
gradients/pi/truediv_grad/Sum_1Sumgradients/pi/truediv_grad/mul1gradients/pi/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
#gradients/pi/truediv_grad/Reshape_1Reshapegradients/pi/truediv_grad/Sum_1!gradients/pi/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
|
*gradients/pi/truediv_grad/tuple/group_depsNoOp"^gradients/pi/truediv_grad/Reshape$^gradients/pi/truediv_grad/Reshape_1
?
2gradients/pi/truediv_grad/tuple/control_dependencyIdentity!gradients/pi/truediv_grad/Reshape+^gradients/pi/truediv_grad/tuple/group_deps*4
_class*
(&loc:@gradients/pi/truediv_grad/Reshape*
T0*'
_output_shapes
:?????????
?
4gradients/pi/truediv_grad/tuple/control_dependency_1Identity#gradients/pi/truediv_grad/Reshape_1+^gradients/pi/truediv_grad/tuple/group_deps*6
_class,
*(loc:@gradients/pi/truediv_grad/Reshape_1*
_output_shapes
:*
T0
h
gradients/pi/sub_grad/ShapeShapePlaceholder_1*
_output_shapes
:*
out_type0*
T0
o
gradients/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
?
+gradients/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/sub_grad/Shapegradients/pi/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/pi/sub_grad/SumSum2gradients/pi/truediv_grad/tuple/control_dependency+gradients/pi/sub_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients/pi/sub_grad/ReshapeReshapegradients/pi/sub_grad/Sumgradients/pi/sub_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients/pi/sub_grad/Sum_1Sum2gradients/pi/truediv_grad/tuple/control_dependency-gradients/pi/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
`
gradients/pi/sub_grad/NegNeggradients/pi/sub_grad/Sum_1*
_output_shapes
:*
T0
?
gradients/pi/sub_grad/Reshape_1Reshapegradients/pi/sub_grad/Neggradients/pi/sub_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0
p
&gradients/pi/sub_grad/tuple/group_depsNoOp^gradients/pi/sub_grad/Reshape ^gradients/pi/sub_grad/Reshape_1
?
.gradients/pi/sub_grad/tuple/control_dependencyIdentitygradients/pi/sub_grad/Reshape'^gradients/pi/sub_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*0
_class&
$"loc:@gradients/pi/sub_grad/Reshape
?
0gradients/pi/sub_grad/tuple/control_dependency_1Identitygradients/pi/sub_grad/Reshape_1'^gradients/pi/sub_grad/tuple/group_deps*2
_class(
&$loc:@gradients/pi/sub_grad/Reshape_1*
T0*'
_output_shapes
:?????????
g
gradients/pi/add_1_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
b
gradients/pi/add_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
-gradients/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_1_grad/Shapegradients/pi/add_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/pi/add_1_grad/SumSum4gradients/pi/truediv_grad/tuple/control_dependency_1-gradients/pi/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients/pi/add_1_grad/ReshapeReshapegradients/pi/add_1_grad/Sumgradients/pi/add_1_grad/Shape*
Tshape0*
_output_shapes
:*
T0
?
gradients/pi/add_1_grad/Sum_1Sum4gradients/pi/truediv_grad/tuple/control_dependency_1/gradients/pi/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
!gradients/pi/add_1_grad/Reshape_1Reshapegradients/pi/add_1_grad/Sum_1gradients/pi/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
v
(gradients/pi/add_1_grad/tuple/group_depsNoOp ^gradients/pi/add_1_grad/Reshape"^gradients/pi/add_1_grad/Reshape_1
?
0gradients/pi/add_1_grad/tuple/control_dependencyIdentitygradients/pi/add_1_grad/Reshape)^gradients/pi/add_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/add_1_grad/Reshape*
_output_shapes
:
?
2gradients/pi/add_1_grad/tuple/control_dependency_1Identity!gradients/pi/add_1_grad/Reshape_1)^gradients/pi/add_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi/add_1_grad/Reshape_1*
_output_shapes
: 
?
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/pi/sub_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:
?
2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad1^gradients/pi/sub_grad/tuple/control_dependency_1
?
:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/pi/sub_grad/tuple/control_dependency_13^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/sub_grad/Reshape_1*'
_output_shapes
:?????????
?
<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
?
gradients/pi/Exp_1_grad/mulMul0gradients/pi/add_1_grad/tuple/control_dependencypi/Exp_1*
_output_shapes
:*
T0
?
'gradients/pi/dense_2/MatMul_grad/MatMulMatMul:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*
transpose_b(*'
_output_shapes
:?????????@*
transpose_a( *
T0
?
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@*
transpose_b( *
T0
?
1gradients/pi/dense_2/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_2/MatMul_grad/MatMul*^gradients/pi/dense_2/MatMul_grad/MatMul_1
?
9gradients/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/MatMul_grad/MatMul2^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????@*:
_class0
.,loc:@gradients/pi/dense_2/MatMul_grad/MatMul*
T0
?
;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_2/MatMul_grad/MatMul_12^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*<
_class2
0.loc:@gradients/pi/dense_2/MatMul_grad/MatMul_1*
T0
?
gradients/AddN_1AddN1gradients/pi/add_10_grad/tuple/control_dependency2gradients/pi/mul_1_grad/tuple/control_dependency_1gradients/pi/Exp_1_grad/mul*3
_class)
'%loc:@gradients/pi/add_10_grad/Reshape*
_output_shapes
:*
T0*
N
?
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh9gradients/pi/dense_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:?????????@*
T0
?
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
_output_shapes
:@*
data_formatNHWC*
T0
?
2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Tanh_grad/TanhGrad
?
:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Tanh_grad/TanhGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients/pi/dense_1/Tanh_grad/TanhGrad*
T0*'
_output_shapes
:?????????@
?
<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad
?
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????@
?
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
?
1gradients/pi/dense_1/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_1/MatMul_grad/MatMul*^gradients/pi/dense_1/MatMul_grad/MatMul_1
?
9gradients/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/MatMul_grad/MatMul2^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*:
_class0
.,loc:@gradients/pi/dense_1/MatMul_grad/MatMul*'
_output_shapes
:?????????@*
T0
?
;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_1/MatMul_grad/MatMul_12^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients/pi/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@*
T0
?
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh9gradients/pi/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????@
?
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
_output_shapes
:@*
data_formatNHWC*
T0
?
0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Tanh_grad/TanhGrad
?
8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Tanh_grad/TanhGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????@*8
_class.
,*loc:@gradients/pi/dense/Tanh_grad/TanhGrad
?
:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
%gradients/pi/dense/MatMul_grad/MatMulMatMul8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:?????????<*
transpose_b(
?
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:<@
?
/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp&^gradients/pi/dense/MatMul_grad/MatMul(^gradients/pi/dense/MatMul_grad/MatMul_1
?
7gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity%gradients/pi/dense/MatMul_grad/MatMul0^gradients/pi/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????<*
T0*8
_class.
,*loc:@gradients/pi/dense/MatMul_grad/MatMul
?
9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity'gradients/pi/dense/MatMul_grad/MatMul_10^gradients/pi/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:<@*
T0*:
_class0
.,loc:@gradients/pi/dense/MatMul_grad/MatMul_1
`
Reshape/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?
ReshapeReshape9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape/shape*
_output_shapes	
:?*
Tshape0*
T0
b
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
	Reshape_1Reshape:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_1/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_2/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?
	Reshape_2Reshape;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_2/shape*
T0*
Tshape0*
_output_shapes	
:? 
b
Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
	Reshape_3Reshape<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_3/shape*
_output_shapes
:@*
Tshape0*
T0
b
Reshape_4/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?
	Reshape_4Reshape;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
T0*
Tshape0*
_output_shapes	
:?
b
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
	Reshape_5Reshape<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
_output_shapes
:*
Tshape0*
T0
b
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
j
	Reshape_6Reshapegradients/AddN_1Reshape_6/shape*
_output_shapes
:*
T0*
Tshape0
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5	Reshape_6concat/axis*
T0*
N*

Tidx0*
_output_shapes	
:?@
g
PyFuncPyFuncconcat*
_output_shapes	
:?@*
Tout
2*
token
pyfunc_0*
Tin
2
l
Const_4Const*
_output_shapes
:*1
value(B&"   @      @   ?         *
dtype0
Q
split/split_dimConst*
value	B : *
_output_shapes
: *
dtype0
?
splitSplitVPyFuncConst_4split/split_dim*

Tlen0*
T0*
	num_split*A
_output_shapes/
-:?:@:? :@:?::
`
Reshape_7/shapeConst*
dtype0*
valueB"<   @   *
_output_shapes
:
c
	Reshape_7ReshapesplitReshape_7/shape*
Tshape0*
T0*
_output_shapes

:<@
Y
Reshape_8/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
a
	Reshape_8Reshapesplit:1Reshape_8/shape*
Tshape0*
T0*
_output_shapes
:@
`
Reshape_9/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
e
	Reshape_9Reshapesplit:2Reshape_9/shape*
Tshape0*
_output_shapes

:@@*
T0
Z
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
c

Reshape_10Reshapesplit:3Reshape_10/shape*
Tshape0*
_output_shapes
:@*
T0
a
Reshape_11/shapeConst*
dtype0*
valueB"@      *
_output_shapes
:
g

Reshape_11Reshapesplit:4Reshape_11/shape*
Tshape0*
T0*
_output_shapes

:@
Z
Reshape_12/shapeConst*
dtype0*
valueB:*
_output_shapes
:
c

Reshape_12Reshapesplit:5Reshape_12/shape*
T0*
_output_shapes
:*
Tshape0
Z
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB:
c

Reshape_13Reshapesplit:6Reshape_13/shape*
_output_shapes
:*
Tshape0*
T0
?
beta1_power/initial_valueConst*
dtype0*
valueB
 *fff?* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
?
beta1_power
VariableV2*
shape: *
dtype0*
shared_name *
	container * 
_class
loc:@pi/dense/bias*
_output_shapes
: 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
l
beta1_power/readIdentitybeta1_power* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: 
?
beta2_power/initial_valueConst* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
dtype0*
valueB
 *w??
?
beta2_power
VariableV2*
dtype0*
shared_name *
	container *
_output_shapes
: *
shape: * 
_class
loc:@pi/dense/bias
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
l
beta2_power/readIdentitybeta2_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
?
6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"<   @   *"
_class
loc:@pi/dense/kernel*
_output_shapes
:
?
,pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *"
_class
loc:@pi/dense/kernel
?
&pi/dense/kernel/Adam/Initializer/zerosFill6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,pi/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes

:<@*
T0*

index_type0*"
_class
loc:@pi/dense/kernel
?
pi/dense/kernel/Adam
VariableV2*"
_class
loc:@pi/dense/kernel*
	container *
shared_name *
shape
:<@*
dtype0*
_output_shapes

:<@
?
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:<@*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
?
pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:<@
?
8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"<   @   *"
_class
loc:@pi/dense/kernel*
dtype0
?
.pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: *"
_class
loc:@pi/dense/kernel
?
(pi/dense/kernel/Adam_1/Initializer/zerosFill8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.pi/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*"
_class
loc:@pi/dense/kernel*

index_type0*
_output_shapes

:<@
?
pi/dense/kernel/Adam_1
VariableV2*
	container *
shape
:<@*
_output_shapes

:<@*"
_class
loc:@pi/dense/kernel*
dtype0*
shared_name 
?
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0*
_output_shapes

:<@*"
_class
loc:@pi/dense/kernel*
use_locking(
?
pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:<@
?
$pi/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
:@*
dtype0*
valueB@*    * 
_class
loc:@pi/dense/bias
?
pi/dense/bias/Adam
VariableV2*
shared_name *
dtype0* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
shape:@*
	container 
?
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(* 
_class
loc:@pi/dense/bias
~
pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias
?
&pi/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
valueB@*    *
dtype0
?
pi/dense/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:@*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
	container 
?
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
:@
?
pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias
?
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*$
_class
loc:@pi/dense_1/kernel*
dtype0*
valueB"@   @   
?
.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0
?
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
T0
?
pi/dense_1/kernel/Adam
VariableV2*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
	container *
shared_name *
shape
:@@*
dtype0
?
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
?
pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
?
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
:*
valueB"@   @   
?
0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    
?
*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*$
_class
loc:@pi/dense_1/kernel*

index_type0*
_output_shapes

:@@*
T0
?
pi/dense_1/kernel/Adam_1
VariableV2*
	container *
shape
:@@*
dtype0*
shared_name *$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
?
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@*
use_locking(
?
pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*
T0*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel
?
&pi/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
valueB@*    *
dtype0
?
pi/dense_1/bias/Adam
VariableV2*
	container *
shape:@*
shared_name *"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
dtype0
?
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
?
pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
?
(pi/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *
dtype0*"
_class
loc:@pi/dense_1/bias
?
pi/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
dtype0*
shared_name *
	container *
shape:@
?
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
?
pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
?
(pi/dense_2/kernel/Adam/Initializer/zerosConst*
valueB@*    *$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes

:@
?
pi/dense_2/kernel/Adam
VariableV2*
_output_shapes

:@*
shared_name *
dtype0*$
_class
loc:@pi/dense_2/kernel*
	container *
shape
:@
?
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
?
pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
T0
?
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*$
_class
loc:@pi/dense_2/kernel*
valueB@*    *
_output_shapes

:@
?
pi/dense_2/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@pi/dense_2/kernel*
shape
:@*
_output_shapes

:@*
	container *
dtype0
?
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(
?
pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:@
?
&pi/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0
?
pi/dense_2/bias/Adam
VariableV2*
shared_name *
_output_shapes
:*
dtype0*"
_class
loc:@pi/dense_2/bias*
shape:*
	container 
?
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
?
pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
?
(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0
?
pi/dense_2/bias/Adam_1
VariableV2*
shape:*"
_class
loc:@pi/dense_2/bias*
	container *
shared_name *
dtype0*
_output_shapes
:
?
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
?
pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
?
!pi/log_std/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@pi/log_std
?
pi/log_std/Adam
VariableV2*
shape:*
_output_shapes
:*
dtype0*
shared_name *
_class
loc:@pi/log_std*
	container 
?
pi/log_std/Adam/AssignAssignpi/log_std/Adam!pi/log_std/Adam/Initializer/zeros*
T0*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(*
validate_shape(
u
pi/log_std/Adam/readIdentitypi/log_std/Adam*
T0*
_class
loc:@pi/log_std*
_output_shapes
:
?
#pi/log_std/Adam_1/Initializer/zerosConst*
_class
loc:@pi/log_std*
valueB*    *
dtype0*
_output_shapes
:
?
pi/log_std/Adam_1
VariableV2*
	container *
shared_name *
_class
loc:@pi/log_std*
dtype0*
_output_shapes
:*
shape:
?
pi/log_std/Adam_1/AssignAssignpi/log_std/Adam_1#pi/log_std/Adam_1/Initializer/zeros*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(
y
pi/log_std/Adam_1/readIdentitypi/log_std/Adam_1*
T0*
_output_shapes
:*
_class
loc:@pi/log_std
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w??*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w?+2*
_output_shapes
: *
dtype0
?
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_7*
use_locking( *"
_class
loc:@pi/dense/kernel*
_output_shapes

:<@*
T0*
use_nesterov( 
?
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_8*
use_locking( *
T0*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_nesterov( 
?
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_9*$
_class
loc:@pi/dense_1/kernel*
use_locking( *
_output_shapes

:@@*
T0*
use_nesterov( 
?
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_10*
use_locking( *
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias*
use_nesterov( 
?
'Adam/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_11*
T0*
use_locking( *
use_nesterov( *$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
?
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_12*
T0*"
_class
loc:@pi/dense_2/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
?
 Adam/update_pi/log_std/ApplyAdam	ApplyAdam
pi/log_stdpi/log_std/Adampi/log_std/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_13*
use_nesterov( *
_output_shapes
:*
T0*
use_locking( *
_class
loc:@pi/log_std
?
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1* 
_class
loc:@pi/dense/bias*
use_locking( *
_output_shapes
: *
validate_shape(*
T0
?
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam!^Adam/update_pi/log_std/ApplyAdam
j
Reshape_14/shapeConst^Adam*
valueB:
?????????*
dtype0*
_output_shapes
:
q

Reshape_14Reshapepi/dense/kernel/readReshape_14/shape*
_output_shapes	
:?*
T0*
Tshape0
j
Reshape_15/shapeConst^Adam*
valueB:
?????????*
dtype0*
_output_shapes
:
n

Reshape_15Reshapepi/dense/bias/readReshape_15/shape*
Tshape0*
_output_shapes
:@*
T0
j
Reshape_16/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
?????????
s

Reshape_16Reshapepi/dense_1/kernel/readReshape_16/shape*
_output_shapes	
:? *
Tshape0*
T0
j
Reshape_17/shapeConst^Adam*
_output_shapes
:*
valueB:
?????????*
dtype0
p

Reshape_17Reshapepi/dense_1/bias/readReshape_17/shape*
Tshape0*
T0*
_output_shapes
:@
j
Reshape_18/shapeConst^Adam*
valueB:
?????????*
_output_shapes
:*
dtype0
s

Reshape_18Reshapepi/dense_2/kernel/readReshape_18/shape*
_output_shapes	
:?*
T0*
Tshape0
j
Reshape_19/shapeConst^Adam*
_output_shapes
:*
valueB:
?????????*
dtype0
p

Reshape_19Reshapepi/dense_2/bias/readReshape_19/shape*
Tshape0*
_output_shapes
:*
T0
j
Reshape_20/shapeConst^Adam*
_output_shapes
:*
valueB:
?????????*
dtype0
k

Reshape_20Reshapepi/log_std/readReshape_20/shape*
_output_shapes
:*
Tshape0*
T0
V
concat_1/axisConst^Adam*
_output_shapes
: *
value	B : *
dtype0
?
concat_1ConcatV2
Reshape_14
Reshape_15
Reshape_16
Reshape_17
Reshape_18
Reshape_19
Reshape_20concat_1/axis*
_output_shapes	
:?@*
T0*
N*

Tidx0
h
PyFunc_1PyFuncconcat_1*
Tin
2*
Tout
2*
_output_shapes
:*
token
pyfunc_1
s
Const_5Const^Adam*
dtype0*
_output_shapes
:*1
value(B&"   @      @   ?         
Z
split_1/split_dimConst^Adam*
value	B : *
dtype0*
_output_shapes
: 
?
split_1SplitVPyFunc_1Const_5split_1/split_dim*0
_output_shapes
:::::::*
T0*

Tlen0*
	num_split
h
Reshape_21/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB"<   @   
g

Reshape_21Reshapesplit_1Reshape_21/shape*
T0*
Tshape0*
_output_shapes

:<@
a
Reshape_22/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:@
e

Reshape_22Reshape	split_1:1Reshape_22/shape*
T0*
Tshape0*
_output_shapes
:@
h
Reshape_23/shapeConst^Adam*
_output_shapes
:*
valueB"@   @   *
dtype0
i

Reshape_23Reshape	split_1:2Reshape_23/shape*
T0*
_output_shapes

:@@*
Tshape0
a
Reshape_24/shapeConst^Adam*
valueB:@*
_output_shapes
:*
dtype0
e

Reshape_24Reshape	split_1:3Reshape_24/shape*
Tshape0*
T0*
_output_shapes
:@
h
Reshape_25/shapeConst^Adam*
valueB"@      *
dtype0*
_output_shapes
:
i

Reshape_25Reshape	split_1:4Reshape_25/shape*
_output_shapes

:@*
Tshape0*
T0
a
Reshape_26/shapeConst^Adam*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_26Reshape	split_1:5Reshape_26/shape*
Tshape0*
_output_shapes
:*
T0
a
Reshape_27/shapeConst^Adam*
dtype0*
valueB:*
_output_shapes
:
e

Reshape_27Reshape	split_1:6Reshape_27/shape*
_output_shapes
:*
T0*
Tshape0
?
AssignAssignpi/dense/kernel
Reshape_21*
_output_shapes

:<@*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0
?
Assign_1Assignpi/dense/bias
Reshape_22*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
?
Assign_2Assignpi/dense_1/kernel
Reshape_23*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

:@@*
use_locking(
?
Assign_3Assignpi/dense_1/bias
Reshape_24*
use_locking(*
validate_shape(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0
?
Assign_4Assignpi/dense_2/kernel
Reshape_25*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
?
Assign_5Assignpi/dense_2/bias
Reshape_26*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
?
Assign_6Assign
pi/log_std
Reshape_27*
T0*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std*
use_locking(
d

group_depsNoOp^Adam^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6
(
group_deps_1NoOp^Adam^group_deps
]
Variable_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
n

Variable_1
VariableV2*
shape: *
dtype0*
shared_name *
	container *
_output_shapes
: 
?
Variable_1/AssignAssign
Variable_1Variable_1/initial_value*
validate_shape(*
_class
loc:@Variable_1*
T0*
use_locking(*
_output_shapes
: 
g
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes
: *
T0
d
PolynomialDecay_1/learning_rateConst*
valueB
 *??8*
_output_shapes
: *
dtype0
]
PolynomialDecay_1/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *?7?5
_
PolynomialDecay_1/Cast_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
]
PolynomialDecay_1/Cast_3/xConst*
value
B :?*
_output_shapes
: *
dtype0
|
PolynomialDecay_1/Cast_3CastPolynomialDecay_1/Cast_3/x*
_output_shapes
: *

SrcT0*
Truncate( *

DstT0
`
PolynomialDecay_1/Minimum/yConst*
_output_shapes
: *
valueB
 *  zE*
dtype0
s
PolynomialDecay_1/MinimumMinimumVariable_1/readPolynomialDecay_1/Minimum/y*
T0*
_output_shapes
: 
v
PolynomialDecay_1/divRealDivPolynomialDecay_1/MinimumPolynomialDecay_1/Cast_3*
_output_shapes
: *
T0
x
PolynomialDecay_1/subSubPolynomialDecay_1/learning_ratePolynomialDecay_1/Cast/x*
T0*
_output_shapes
: 
^
PolynomialDecay_1/sub_1/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
q
PolynomialDecay_1/sub_1SubPolynomialDecay_1/sub_1/xPolynomialDecay_1/div*
_output_shapes
: *
T0
r
PolynomialDecay_1/PowPowPolynomialDecay_1/sub_1PolynomialDecay_1/Cast_1/x*
_output_shapes
: *
T0
k
PolynomialDecay_1/MulMulPolynomialDecay_1/subPolynomialDecay_1/Pow*
_output_shapes
: *
T0
j
PolynomialDecay_1AddPolynomialDecay_1/MulPolynomialDecay_1/Cast/x*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_1/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
T
gradients_1/Neg_2_grad/NegNeggradients_1/Fill*
_output_shapes
: *
T0
o
%gradients_1/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
?
gradients_1/Mean_3_grad/ReshapeReshapegradients_1/Neg_2_grad/Neg%gradients_1/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_1/Mean_3_grad/ShapeShapeMul_5*
T0*
_output_shapes
:*
out_type0
?
gradients_1/Mean_3_grad/TileTilegradients_1/Mean_3_grad/Reshapegradients_1/Mean_3_grad/Shape*#
_output_shapes
:?????????*

Tmultiples0*
T0
d
gradients_1/Mean_3_grad/Shape_1ShapeMul_5*
_output_shapes
:*
out_type0*
T0
b
gradients_1/Mean_3_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_1/Mean_3_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
?
gradients_1/Mean_3_grad/ProdProdgradients_1/Mean_3_grad/Shape_1gradients_1/Mean_3_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
i
gradients_1/Mean_3_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_1/Mean_3_grad/Prod_1Prodgradients_1/Mean_3_grad/Shape_2gradients_1/Mean_3_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_1/Mean_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
?
gradients_1/Mean_3_grad/MaximumMaximumgradients_1/Mean_3_grad/Prod_1!gradients_1/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0
?
 gradients_1/Mean_3_grad/floordivFloorDivgradients_1/Mean_3_grad/Prodgradients_1/Mean_3_grad/Maximum*
_output_shapes
: *
T0
?
gradients_1/Mean_3_grad/CastCast gradients_1/Mean_3_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0*
Truncate( 
?
gradients_1/Mean_3_grad/truedivRealDivgradients_1/Mean_3_grad/Tilegradients_1/Mean_3_grad/Cast*#
_output_shapes
:?????????*
T0
g
gradients_1/Mul_5_grad/ShapeShapelam/Squeeze*
T0*
out_type0*
_output_shapes
:
l
gradients_1/Mul_5_grad/Shape_1ShapeStopGradient_1*
T0*
out_type0*
_output_shapes
:
?
,gradients_1/Mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Mul_5_grad/Shapegradients_1/Mul_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/Mul_5_grad/MulMulgradients_1/Mean_3_grad/truedivStopGradient_1*#
_output_shapes
:?????????*
T0
?
gradients_1/Mul_5_grad/SumSumgradients_1/Mul_5_grad/Mul,gradients_1/Mul_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients_1/Mul_5_grad/ReshapeReshapegradients_1/Mul_5_grad/Sumgradients_1/Mul_5_grad/Shape*
Tshape0*
T0*#
_output_shapes
:?????????

gradients_1/Mul_5_grad/Mul_1Mullam/Squeezegradients_1/Mean_3_grad/truediv*
T0*#
_output_shapes
:?????????
?
gradients_1/Mul_5_grad/Sum_1Sumgradients_1/Mul_5_grad/Mul_1.gradients_1/Mul_5_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
 gradients_1/Mul_5_grad/Reshape_1Reshapegradients_1/Mul_5_grad/Sum_1gradients_1/Mul_5_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:?????????
s
'gradients_1/Mul_5_grad/tuple/group_depsNoOp^gradients_1/Mul_5_grad/Reshape!^gradients_1/Mul_5_grad/Reshape_1
?
/gradients_1/Mul_5_grad/tuple/control_dependencyIdentitygradients_1/Mul_5_grad/Reshape(^gradients_1/Mul_5_grad/tuple/group_deps*#
_output_shapes
:?????????*1
_class'
%#loc:@gradients_1/Mul_5_grad/Reshape*
T0
?
1gradients_1/Mul_5_grad/tuple/control_dependency_1Identity gradients_1/Mul_5_grad/Reshape_1(^gradients_1/Mul_5_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*3
_class)
'%loc:@gradients_1/Mul_5_grad/Reshape_1
v
"gradients_1/lam/Squeeze_grad/ShapeShapelam/dense_2/Softplus*
out_type0*
T0*
_output_shapes
:
?
$gradients_1/lam/Squeeze_grad/ReshapeReshape/gradients_1/Mul_5_grad/tuple/control_dependency"gradients_1/lam/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????
?
2gradients_1/lam/dense_2/Softplus_grad/SoftplusGradSoftplusGrad$gradients_1/lam/Squeeze_grad/Reshapelam/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
?
0gradients_1/lam/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients_1/lam/dense_2/Softplus_grad/SoftplusGrad*
_output_shapes
:*
T0*
data_formatNHWC
?
5gradients_1/lam/dense_2/BiasAdd_grad/tuple/group_depsNoOp1^gradients_1/lam/dense_2/BiasAdd_grad/BiasAddGrad3^gradients_1/lam/dense_2/Softplus_grad/SoftplusGrad
?
=gradients_1/lam/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity2gradients_1/lam/dense_2/Softplus_grad/SoftplusGrad6^gradients_1/lam/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????*E
_class;
97loc:@gradients_1/lam/dense_2/Softplus_grad/SoftplusGrad*
T0
?
?gradients_1/lam/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_1/lam/dense_2/BiasAdd_grad/BiasAddGrad6^gradients_1/lam/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*C
_class9
75loc:@gradients_1/lam/dense_2/BiasAdd_grad/BiasAddGrad
?
*gradients_1/lam/dense_2/MatMul_grad/MatMulMatMul=gradients_1/lam/dense_2/BiasAdd_grad/tuple/control_dependencylam/dense_2/kernel/read*
T0*'
_output_shapes
:?????????@*
transpose_b(*
transpose_a( 
?
,gradients_1/lam/dense_2/MatMul_grad/MatMul_1MatMullam/dense_1/Tanh=gradients_1/lam/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
?
4gradients_1/lam/dense_2/MatMul_grad/tuple/group_depsNoOp+^gradients_1/lam/dense_2/MatMul_grad/MatMul-^gradients_1/lam/dense_2/MatMul_grad/MatMul_1
?
<gradients_1/lam/dense_2/MatMul_grad/tuple/control_dependencyIdentity*gradients_1/lam/dense_2/MatMul_grad/MatMul5^gradients_1/lam/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????@*=
_class3
1/loc:@gradients_1/lam/dense_2/MatMul_grad/MatMul*
T0
?
>gradients_1/lam/dense_2/MatMul_grad/tuple/control_dependency_1Identity,gradients_1/lam/dense_2/MatMul_grad/MatMul_15^gradients_1/lam/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*?
_class5
31loc:@gradients_1/lam/dense_2/MatMul_grad/MatMul_1
?
*gradients_1/lam/dense_1/Tanh_grad/TanhGradTanhGradlam/dense_1/Tanh<gradients_1/lam/dense_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:?????????@*
T0
?
0gradients_1/lam/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients_1/lam/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes
:@
?
5gradients_1/lam/dense_1/BiasAdd_grad/tuple/group_depsNoOp1^gradients_1/lam/dense_1/BiasAdd_grad/BiasAddGrad+^gradients_1/lam/dense_1/Tanh_grad/TanhGrad
?
=gradients_1/lam/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity*gradients_1/lam/dense_1/Tanh_grad/TanhGrad6^gradients_1/lam/dense_1/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????@*=
_class3
1/loc:@gradients_1/lam/dense_1/Tanh_grad/TanhGrad
?
?gradients_1/lam/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity0gradients_1/lam/dense_1/BiasAdd_grad/BiasAddGrad6^gradients_1/lam/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*C
_class9
75loc:@gradients_1/lam/dense_1/BiasAdd_grad/BiasAddGrad*
T0
?
*gradients_1/lam/dense_1/MatMul_grad/MatMulMatMul=gradients_1/lam/dense_1/BiasAdd_grad/tuple/control_dependencylam/dense_1/kernel/read*
T0*'
_output_shapes
:?????????@*
transpose_b(*
transpose_a( 
?
,gradients_1/lam/dense_1/MatMul_grad/MatMul_1MatMullam/dense/Tanh=gradients_1/lam/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@@*
transpose_a(
?
4gradients_1/lam/dense_1/MatMul_grad/tuple/group_depsNoOp+^gradients_1/lam/dense_1/MatMul_grad/MatMul-^gradients_1/lam/dense_1/MatMul_grad/MatMul_1
?
<gradients_1/lam/dense_1/MatMul_grad/tuple/control_dependencyIdentity*gradients_1/lam/dense_1/MatMul_grad/MatMul5^gradients_1/lam/dense_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/lam/dense_1/MatMul_grad/MatMul*'
_output_shapes
:?????????@
?
>gradients_1/lam/dense_1/MatMul_grad/tuple/control_dependency_1Identity,gradients_1/lam/dense_1/MatMul_grad/MatMul_15^gradients_1/lam/dense_1/MatMul_grad/tuple/group_deps*
_output_shapes

:@@*?
_class5
31loc:@gradients_1/lam/dense_1/MatMul_grad/MatMul_1*
T0
?
(gradients_1/lam/dense/Tanh_grad/TanhGradTanhGradlam/dense/Tanh<gradients_1/lam/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????@
?
.gradients_1/lam/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/lam/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
?
3gradients_1/lam/dense/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/lam/dense/BiasAdd_grad/BiasAddGrad)^gradients_1/lam/dense/Tanh_grad/TanhGrad
?
;gradients_1/lam/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/lam/dense/Tanh_grad/TanhGrad4^gradients_1/lam/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/lam/dense/Tanh_grad/TanhGrad*'
_output_shapes
:?????????@
?
=gradients_1/lam/dense/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/lam/dense/BiasAdd_grad/BiasAddGrad4^gradients_1/lam/dense/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/lam/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
(gradients_1/lam/dense/MatMul_grad/MatMulMatMul;gradients_1/lam/dense/BiasAdd_grad/tuple/control_dependencylam/dense/kernel/read*'
_output_shapes
:?????????<*
T0*
transpose_a( *
transpose_b(
?
*gradients_1/lam/dense/MatMul_grad/MatMul_1MatMulPlaceholder;gradients_1/lam/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes

:<@*
T0
?
2gradients_1/lam/dense/MatMul_grad/tuple/group_depsNoOp)^gradients_1/lam/dense/MatMul_grad/MatMul+^gradients_1/lam/dense/MatMul_grad/MatMul_1
?
:gradients_1/lam/dense/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/lam/dense/MatMul_grad/MatMul3^gradients_1/lam/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????<*
T0*;
_class1
/-loc:@gradients_1/lam/dense/MatMul_grad/MatMul
?
<gradients_1/lam/dense/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/lam/dense/MatMul_grad/MatMul_13^gradients_1/lam/dense/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_1/lam/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes

:<@
c
Reshape_28/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_28Reshape<gradients_1/lam/dense/MatMul_grad/tuple/control_dependency_1Reshape_28/shape*
T0*
Tshape0*
_output_shapes	
:?
c
Reshape_29/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?

Reshape_29Reshape=gradients_1/lam/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_29/shape*
_output_shapes
:@*
Tshape0*
T0
c
Reshape_30/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?

Reshape_30Reshape>gradients_1/lam/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_30/shape*
T0*
Tshape0*
_output_shapes	
:? 
c
Reshape_31/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_31Reshape?gradients_1/lam/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_31/shape*
T0*
_output_shapes
:@*
Tshape0
c
Reshape_32/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?

Reshape_32Reshape>gradients_1/lam/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_32/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_33/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?

Reshape_33Reshape?gradients_1/lam/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_33/shape*
T0*
_output_shapes
:*
Tshape0
O
concat_2/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
concat_2ConcatV2
Reshape_28
Reshape_29
Reshape_30
Reshape_31
Reshape_32
Reshape_33concat_2/axis*

Tidx0*
_output_shapes	
:??*
N*
T0
k
PyFunc_2PyFuncconcat_2*
_output_shapes	
:??*
token
pyfunc_2*
Tin
2*
Tout
2
h
Const_6Const*
_output_shapes
:*
dtype0*-
value$B""   @      @   @      
S
split_2/split_dimConst*
value	B : *
_output_shapes
: *
dtype0
?
split_2SplitVPyFunc_2Const_6split_2/split_dim*
T0*

Tlen0*:
_output_shapes(
&:?:@:? :@:@:*
	num_split
a
Reshape_34/shapeConst*
valueB"<   @   *
dtype0*
_output_shapes
:
g

Reshape_34Reshapesplit_2Reshape_34/shape*
T0*
Tshape0*
_output_shapes

:<@
Z
Reshape_35/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
e

Reshape_35Reshape	split_2:1Reshape_35/shape*
_output_shapes
:@*
T0*
Tshape0
a
Reshape_36/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
i

Reshape_36Reshape	split_2:2Reshape_36/shape*
T0*
Tshape0*
_output_shapes

:@@
Z
Reshape_37/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
e

Reshape_37Reshape	split_2:3Reshape_37/shape*
_output_shapes
:@*
T0*
Tshape0
a
Reshape_38/shapeConst*
dtype0*
_output_shapes
:*
valueB"@      
i

Reshape_38Reshape	split_2:4Reshape_38/shape*
Tshape0*
T0*
_output_shapes

:@
Z
Reshape_39/shapeConst*
_output_shapes
:*
valueB:*
dtype0
e

Reshape_39Reshape	split_2:5Reshape_39/shape*
T0*
_output_shapes
:*
Tshape0
?
beta1_power_1/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: *!
_class
loc:@lam/dense/bias
?
beta1_power_1
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
shape: *!
_class
loc:@lam/dense/bias*
	container 
?
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *!
_class
loc:@lam/dense/bias*
T0*
use_locking(
q
beta1_power_1/readIdentitybeta1_power_1*!
_class
loc:@lam/dense/bias*
_output_shapes
: *
T0
?
beta2_power_1/initial_valueConst*
_output_shapes
: *!
_class
loc:@lam/dense/bias*
valueB
 *w??*
dtype0
?
beta2_power_1
VariableV2*
shared_name *
	container *
_output_shapes
: *
shape: *
dtype0*!
_class
loc:@lam/dense/bias
?
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
_output_shapes
: *!
_class
loc:@lam/dense/bias*
use_locking(*
T0
q
beta2_power_1/readIdentitybeta2_power_1*!
_class
loc:@lam/dense/bias*
_output_shapes
: *
T0
?
7lam/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"<   @   *#
_class
loc:@lam/dense/kernel*
_output_shapes
:*
dtype0
?
-lam/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *#
_class
loc:@lam/dense/kernel*
dtype0
?
'lam/dense/kernel/Adam/Initializer/zerosFill7lam/dense/kernel/Adam/Initializer/zeros/shape_as_tensor-lam/dense/kernel/Adam/Initializer/zeros/Const*

index_type0*#
_class
loc:@lam/dense/kernel*
T0*
_output_shapes

:<@
?
lam/dense/kernel/Adam
VariableV2*
_output_shapes

:<@*
	container *
dtype0*
shared_name *
shape
:<@*#
_class
loc:@lam/dense/kernel
?
lam/dense/kernel/Adam/AssignAssignlam/dense/kernel/Adam'lam/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:<@*
validate_shape(*
T0*#
_class
loc:@lam/dense/kernel*
use_locking(
?
lam/dense/kernel/Adam/readIdentitylam/dense/kernel/Adam*#
_class
loc:@lam/dense/kernel*
T0*
_output_shapes

:<@
?
9lam/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*#
_class
loc:@lam/dense/kernel*
valueB"<   @   
?
/lam/dense/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@lam/dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *    
?
)lam/dense/kernel/Adam_1/Initializer/zerosFill9lam/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor/lam/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:<@*

index_type0*#
_class
loc:@lam/dense/kernel*
T0
?
lam/dense/kernel/Adam_1
VariableV2*
	container *#
_class
loc:@lam/dense/kernel*
_output_shapes

:<@*
shared_name *
shape
:<@*
dtype0
?
lam/dense/kernel/Adam_1/AssignAssignlam/dense/kernel/Adam_1)lam/dense/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes

:<@*
validate_shape(*#
_class
loc:@lam/dense/kernel*
use_locking(
?
lam/dense/kernel/Adam_1/readIdentitylam/dense/kernel/Adam_1*
T0*#
_class
loc:@lam/dense/kernel*
_output_shapes

:<@
?
%lam/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *!
_class
loc:@lam/dense/bias
?
lam/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *!
_class
loc:@lam/dense/bias*
shape:@*
	container 
?
lam/dense/bias/Adam/AssignAssignlam/dense/bias/Adam%lam/dense/bias/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@lam/dense/bias*
validate_shape(
?
lam/dense/bias/Adam/readIdentitylam/dense/bias/Adam*
T0*
_output_shapes
:@*!
_class
loc:@lam/dense/bias
?
'lam/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*!
_class
loc:@lam/dense/bias*
valueB@*    *
_output_shapes
:@
?
lam/dense/bias/Adam_1
VariableV2*
dtype0*!
_class
loc:@lam/dense/bias*
shape:@*
	container *
shared_name *
_output_shapes
:@
?
lam/dense/bias/Adam_1/AssignAssignlam/dense/bias/Adam_1'lam/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:@*!
_class
loc:@lam/dense/bias*
T0*
validate_shape(
?
lam/dense/bias/Adam_1/readIdentitylam/dense/bias/Adam_1*
T0*!
_class
loc:@lam/dense/bias*
_output_shapes
:@
?
9lam/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *%
_class
loc:@lam/dense_1/kernel*
_output_shapes
:*
dtype0
?
/lam/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *%
_class
loc:@lam/dense_1/kernel*
dtype0
?
)lam/dense_1/kernel/Adam/Initializer/zerosFill9lam/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor/lam/dense_1/kernel/Adam/Initializer/zeros/Const*
_output_shapes

:@@*

index_type0*%
_class
loc:@lam/dense_1/kernel*
T0
?
lam/dense_1/kernel/Adam
VariableV2*
_output_shapes

:@@*
dtype0*
shared_name *
	container *%
_class
loc:@lam/dense_1/kernel*
shape
:@@
?
lam/dense_1/kernel/Adam/AssignAssignlam/dense_1/kernel/Adam)lam/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*%
_class
loc:@lam/dense_1/kernel*
_output_shapes

:@@
?
lam/dense_1/kernel/Adam/readIdentitylam/dense_1/kernel/Adam*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel*
T0
?
;lam/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"@   @   *
_output_shapes
:*%
_class
loc:@lam/dense_1/kernel
?
1lam/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: *%
_class
loc:@lam/dense_1/kernel
?
+lam/dense_1/kernel/Adam_1/Initializer/zerosFill;lam/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor1lam/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel
?
lam/dense_1/kernel/Adam_1
VariableV2*
shape
:@@*
	container *
_output_shapes

:@@*
dtype0*
shared_name *%
_class
loc:@lam/dense_1/kernel
?
 lam/dense_1/kernel/Adam_1/AssignAssignlam/dense_1/kernel/Adam_1+lam/dense_1/kernel/Adam_1/Initializer/zeros*
T0*%
_class
loc:@lam/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
validate_shape(
?
lam/dense_1/kernel/Adam_1/readIdentitylam/dense_1/kernel/Adam_1*
_output_shapes

:@@*
T0*%
_class
loc:@lam/dense_1/kernel
?
'lam/dense_1/bias/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes
:@*#
_class
loc:@lam/dense_1/bias
?
lam/dense_1/bias/Adam
VariableV2*
	container *
dtype0*
_output_shapes
:@*
shape:@*
shared_name *#
_class
loc:@lam/dense_1/bias
?
lam/dense_1/bias/Adam/AssignAssignlam/dense_1/bias/Adam'lam/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes
:@*#
_class
loc:@lam/dense_1/bias*
T0
?
lam/dense_1/bias/Adam/readIdentitylam/dense_1/bias/Adam*
T0*#
_class
loc:@lam/dense_1/bias*
_output_shapes
:@
?
)lam/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB@*    *
_output_shapes
:@*#
_class
loc:@lam/dense_1/bias
?
lam/dense_1/bias/Adam_1
VariableV2*#
_class
loc:@lam/dense_1/bias*
dtype0*
shape:@*
shared_name *
_output_shapes
:@*
	container 
?
lam/dense_1/bias/Adam_1/AssignAssignlam/dense_1/bias/Adam_1)lam/dense_1/bias/Adam_1/Initializer/zeros*#
_class
loc:@lam/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
?
lam/dense_1/bias/Adam_1/readIdentitylam/dense_1/bias/Adam_1*
T0*
_output_shapes
:@*#
_class
loc:@lam/dense_1/bias
?
)lam/dense_2/kernel/Adam/Initializer/zerosConst*
valueB@*    *
_output_shapes

:@*
dtype0*%
_class
loc:@lam/dense_2/kernel
?
lam/dense_2/kernel/Adam
VariableV2*
dtype0*
	container *
shape
:@*
shared_name *
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel
?
lam/dense_2/kernel/Adam/AssignAssignlam/dense_2/kernel/Adam)lam/dense_2/kernel/Adam/Initializer/zeros*%
_class
loc:@lam/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:@*
use_locking(
?
lam/dense_2/kernel/Adam/readIdentitylam/dense_2/kernel/Adam*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel*
T0
?
+lam/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel
?
lam/dense_2/kernel/Adam_1
VariableV2*
	container *
shape
:@*
shared_name *
dtype0*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel
?
 lam/dense_2/kernel/Adam_1/AssignAssignlam/dense_2/kernel/Adam_1+lam/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel*
validate_shape(
?
lam/dense_2/kernel/Adam_1/readIdentitylam/dense_2/kernel/Adam_1*
T0*%
_class
loc:@lam/dense_2/kernel*
_output_shapes

:@
?
'lam/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@lam/dense_2/bias*
_output_shapes
:*
valueB*    
?
lam/dense_2/bias/Adam
VariableV2*
	container *
shape:*#
_class
loc:@lam/dense_2/bias*
_output_shapes
:*
shared_name *
dtype0
?
lam/dense_2/bias/Adam/AssignAssignlam/dense_2/bias/Adam'lam/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*#
_class
loc:@lam/dense_2/bias*
T0
?
lam/dense_2/bias/Adam/readIdentitylam/dense_2/bias/Adam*
_output_shapes
:*
T0*#
_class
loc:@lam/dense_2/bias
?
)lam/dense_2/bias/Adam_1/Initializer/zerosConst*#
_class
loc:@lam/dense_2/bias*
dtype0*
_output_shapes
:*
valueB*    
?
lam/dense_2/bias/Adam_1
VariableV2*
dtype0*#
_class
loc:@lam/dense_2/bias*
_output_shapes
:*
shared_name *
shape:*
	container 
?
lam/dense_2/bias/Adam_1/AssignAssignlam/dense_2/bias/Adam_1)lam/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*#
_class
loc:@lam/dense_2/bias*
_output_shapes
:*
T0
?
lam/dense_2/bias/Adam_1/readIdentitylam/dense_2/bias/Adam_1*#
_class
loc:@lam/dense_2/bias*
T0*
_output_shapes
:
Q
Adam_1/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
Q
Adam_1/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
dtype0*
valueB
 *w?+2*
_output_shapes
: 
?
(Adam_1/update_lam/dense/kernel/ApplyAdam	ApplyAdamlam/dense/kernellam/dense/kernel/Adamlam/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readPolynomialDecay_1Adam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_34*#
_class
loc:@lam/dense/kernel*
use_locking( *
use_nesterov( *
T0*
_output_shapes

:<@
?
&Adam_1/update_lam/dense/bias/ApplyAdam	ApplyAdamlam/dense/biaslam/dense/bias/Adamlam/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readPolynomialDecay_1Adam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_35*
use_locking( *
T0*
use_nesterov( *
_output_shapes
:@*!
_class
loc:@lam/dense/bias
?
*Adam_1/update_lam/dense_1/kernel/ApplyAdam	ApplyAdamlam/dense_1/kernellam/dense_1/kernel/Adamlam/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readPolynomialDecay_1Adam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_36*
use_locking( *
_output_shapes

:@@*
T0*
use_nesterov( *%
_class
loc:@lam/dense_1/kernel
?
(Adam_1/update_lam/dense_1/bias/ApplyAdam	ApplyAdamlam/dense_1/biaslam/dense_1/bias/Adamlam/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readPolynomialDecay_1Adam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_37*
use_locking( *
_output_shapes
:@*#
_class
loc:@lam/dense_1/bias*
T0*
use_nesterov( 
?
*Adam_1/update_lam/dense_2/kernel/ApplyAdam	ApplyAdamlam/dense_2/kernellam/dense_2/kernel/Adamlam/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readPolynomialDecay_1Adam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_38*
use_nesterov( *
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel*
T0*
use_locking( 
?
(Adam_1/update_lam/dense_2/bias/ApplyAdam	ApplyAdamlam/dense_2/biaslam/dense_2/bias/Adamlam/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readPolynomialDecay_1Adam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_39*
_output_shapes
:*
T0*
use_locking( *#
_class
loc:@lam/dense_2/bias*
use_nesterov( 
?

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1'^Adam_1/update_lam/dense/bias/ApplyAdam)^Adam_1/update_lam/dense/kernel/ApplyAdam)^Adam_1/update_lam/dense_1/bias/ApplyAdam+^Adam_1/update_lam/dense_1/kernel/ApplyAdam)^Adam_1/update_lam/dense_2/bias/ApplyAdam+^Adam_1/update_lam/dense_2/kernel/ApplyAdam*
T0*
_output_shapes
: *!
_class
loc:@lam/dense/bias
?
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*!
_class
loc:@lam/dense/bias*
use_locking( *
T0*
_output_shapes
: *
validate_shape(
?
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2'^Adam_1/update_lam/dense/bias/ApplyAdam)^Adam_1/update_lam/dense/kernel/ApplyAdam)^Adam_1/update_lam/dense_1/bias/ApplyAdam+^Adam_1/update_lam/dense_1/kernel/ApplyAdam)^Adam_1/update_lam/dense_2/bias/ApplyAdam+^Adam_1/update_lam/dense_2/kernel/ApplyAdam*!
_class
loc:@lam/dense/bias*
T0*
_output_shapes
: 
?
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*!
_class
loc:@lam/dense/bias*
use_locking( *
_output_shapes
: *
T0*
validate_shape(
?
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1'^Adam_1/update_lam/dense/bias/ApplyAdam)^Adam_1/update_lam/dense/kernel/ApplyAdam)^Adam_1/update_lam/dense_1/bias/ApplyAdam+^Adam_1/update_lam/dense_1/kernel/ApplyAdam)^Adam_1/update_lam/dense_2/bias/ApplyAdam+^Adam_1/update_lam/dense_2/kernel/ApplyAdam
l
Reshape_40/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
?????????
r

Reshape_40Reshapelam/dense/kernel/readReshape_40/shape*
_output_shapes	
:?*
T0*
Tshape0
l
Reshape_41/shapeConst^Adam_1*
dtype0*
valueB:
?????????*
_output_shapes
:
o

Reshape_41Reshapelam/dense/bias/readReshape_41/shape*
Tshape0*
_output_shapes
:@*
T0
l
Reshape_42/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
?????????
t

Reshape_42Reshapelam/dense_1/kernel/readReshape_42/shape*
T0*
_output_shapes	
:? *
Tshape0
l
Reshape_43/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
?????????
q

Reshape_43Reshapelam/dense_1/bias/readReshape_43/shape*
_output_shapes
:@*
T0*
Tshape0
l
Reshape_44/shapeConst^Adam_1*
valueB:
?????????*
_output_shapes
:*
dtype0
s

Reshape_44Reshapelam/dense_2/kernel/readReshape_44/shape*
Tshape0*
T0*
_output_shapes
:@
l
Reshape_45/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:
?????????
q

Reshape_45Reshapelam/dense_2/bias/readReshape_45/shape*
_output_shapes
:*
T0*
Tshape0
X
concat_3/axisConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
?
concat_3ConcatV2
Reshape_40
Reshape_41
Reshape_42
Reshape_43
Reshape_44
Reshape_45concat_3/axis*
T0*
_output_shapes	
:??*
N*

Tidx0
h
PyFunc_3PyFuncconcat_3*
_output_shapes
:*
Tout
2*
Tin
2*
token
pyfunc_3
q
Const_7Const^Adam_1*
_output_shapes
:*
dtype0*-
value$B""   @      @   @      
\
split_3/split_dimConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
?
split_3SplitVPyFunc_3Const_7split_3/split_dim*
T0*,
_output_shapes
::::::*
	num_split*

Tlen0
j
Reshape_46/shapeConst^Adam_1*
valueB"<   @   *
dtype0*
_output_shapes
:
g

Reshape_46Reshapesplit_3Reshape_46/shape*
_output_shapes

:<@*
T0*
Tshape0
c
Reshape_47/shapeConst^Adam_1*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_47Reshape	split_3:1Reshape_47/shape*
_output_shapes
:@*
Tshape0*
T0
j
Reshape_48/shapeConst^Adam_1*
valueB"@   @   *
dtype0*
_output_shapes
:
i

Reshape_48Reshape	split_3:2Reshape_48/shape*
T0*
_output_shapes

:@@*
Tshape0
c
Reshape_49/shapeConst^Adam_1*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_49Reshape	split_3:3Reshape_49/shape*
T0*
Tshape0*
_output_shapes
:@
j
Reshape_50/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB"@      
i

Reshape_50Reshape	split_3:4Reshape_50/shape*
T0*
Tshape0*
_output_shapes

:@
c
Reshape_51/shapeConst^Adam_1*
_output_shapes
:*
dtype0*
valueB:
e

Reshape_51Reshape	split_3:5Reshape_51/shape*
T0*
Tshape0*
_output_shapes
:
?
Assign_7Assignlam/dense/kernel
Reshape_46*
T0*
use_locking(*
_output_shapes

:<@*
validate_shape(*#
_class
loc:@lam/dense/kernel
?
Assign_8Assignlam/dense/bias
Reshape_47*
use_locking(*
_output_shapes
:@*
T0*!
_class
loc:@lam/dense/bias*
validate_shape(
?
Assign_9Assignlam/dense_1/kernel
Reshape_48*
_output_shapes

:@@*
validate_shape(*
T0*%
_class
loc:@lam/dense_1/kernel*
use_locking(
?
	Assign_10Assignlam/dense_1/bias
Reshape_49*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*#
_class
loc:@lam/dense_1/bias
?
	Assign_11Assignlam/dense_2/kernel
Reshape_50*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*%
_class
loc:@lam/dense_2/kernel
?
	Assign_12Assignlam/dense_2/bias
Reshape_51*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*#
_class
loc:@lam/dense_2/bias
b
group_deps_2NoOp^Adam_1
^Assign_10
^Assign_11
^Assign_12	^Assign_7	^Assign_8	^Assign_9
,
group_deps_3NoOp^Adam_1^group_deps_2
U
sub_2SubPlaceholder_4
vf/Squeeze*
T0*#
_output_shapes
:?????????
J
pow/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
F
powPowsub_2pow/y*
T0*#
_output_shapes
:?????????
Q
Const_8Const*
valueB: *
_output_shapes
:*
dtype0
Z
Mean_4MeanpowConst_8*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
sub_3SubPlaceholder_5
vc/Squeeze*#
_output_shapes
:?????????*
T0
L
pow_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
J
pow_1Powsub_3pow_1/y*#
_output_shapes
:?????????*
T0
Q
Const_9Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_5Meanpow_1Const_9*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
=
add_1AddMean_4Mean_5*
T0*
_output_shapes
: 
]
Variable_2/initial_valueConst*
valueB
 *    *
_output_shapes
: *
dtype0
n

Variable_2
VariableV2*
	container *
_output_shapes
: *
dtype0*
shared_name *
shape: 
?
Variable_2/AssignAssign
Variable_2Variable_2/initial_value*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@Variable_2*
use_locking(
g
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
: 
d
PolynomialDecay_2/learning_rateConst*
_output_shapes
: *
valueB
 *RI?9*
dtype0
]
PolynomialDecay_2/Cast/xConst*
valueB
 *?7?5*
_output_shapes
: *
dtype0
_
PolynomialDecay_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
]
PolynomialDecay_2/Cast_3/xConst*
value
B :?]*
dtype0*
_output_shapes
: 
|
PolynomialDecay_2/Cast_3CastPolynomialDecay_2/Cast_3/x*

DstT0*

SrcT0*
_output_shapes
: *
Truncate( 
`
PolynomialDecay_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?;F
s
PolynomialDecay_2/MinimumMinimumVariable_2/readPolynomialDecay_2/Minimum/y*
T0*
_output_shapes
: 
v
PolynomialDecay_2/divRealDivPolynomialDecay_2/MinimumPolynomialDecay_2/Cast_3*
_output_shapes
: *
T0
x
PolynomialDecay_2/subSubPolynomialDecay_2/learning_ratePolynomialDecay_2/Cast/x*
T0*
_output_shapes
: 
^
PolynomialDecay_2/sub_1/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
q
PolynomialDecay_2/sub_1SubPolynomialDecay_2/sub_1/xPolynomialDecay_2/div*
_output_shapes
: *
T0
r
PolynomialDecay_2/PowPowPolynomialDecay_2/sub_1PolynomialDecay_2/Cast_1/x*
_output_shapes
: *
T0
k
PolynomialDecay_2/MulMulPolynomialDecay_2/subPolynomialDecay_2/Pow*
_output_shapes
: *
T0
j
PolynomialDecay_2AddPolynomialDecay_2/MulPolynomialDecay_2/Cast/x*
_output_shapes
: *
T0
T
gradients_2/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_2/grad_ys_0Const*
dtype0*
valueB
 *  ??*
_output_shapes
: 
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
_output_shapes
: *

index_type0*
T0
B
'gradients_2/add_1_grad/tuple/group_depsNoOp^gradients_2/Fill
?
/gradients_2/add_1_grad/tuple/control_dependencyIdentitygradients_2/Fill(^gradients_2/add_1_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_2/Fill
?
1gradients_2/add_1_grad/tuple/control_dependency_1Identitygradients_2/Fill(^gradients_2/add_1_grad/tuple/group_deps*#
_class
loc:@gradients_2/Fill*
T0*
_output_shapes
: 
o
%gradients_2/Mean_4_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
?
gradients_2/Mean_4_grad/ReshapeReshape/gradients_2/add_1_grad/tuple/control_dependency%gradients_2/Mean_4_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
`
gradients_2/Mean_4_grad/ShapeShapepow*
T0*
out_type0*
_output_shapes
:
?
gradients_2/Mean_4_grad/TileTilegradients_2/Mean_4_grad/Reshapegradients_2/Mean_4_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:?????????
b
gradients_2/Mean_4_grad/Shape_1Shapepow*
_output_shapes
:*
out_type0*
T0
b
gradients_2/Mean_4_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_2/Mean_4_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_2/Mean_4_grad/ProdProdgradients_2/Mean_4_grad/Shape_1gradients_2/Mean_4_grad/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
i
gradients_2/Mean_4_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
gradients_2/Mean_4_grad/Prod_1Prodgradients_2/Mean_4_grad/Shape_2gradients_2/Mean_4_grad/Const_1*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
c
!gradients_2/Mean_4_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
gradients_2/Mean_4_grad/MaximumMaximumgradients_2/Mean_4_grad/Prod_1!gradients_2/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0
?
 gradients_2/Mean_4_grad/floordivFloorDivgradients_2/Mean_4_grad/Prodgradients_2/Mean_4_grad/Maximum*
_output_shapes
: *
T0
?
gradients_2/Mean_4_grad/CastCast gradients_2/Mean_4_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0*
Truncate( 
?
gradients_2/Mean_4_grad/truedivRealDivgradients_2/Mean_4_grad/Tilegradients_2/Mean_4_grad/Cast*
T0*#
_output_shapes
:?????????
o
%gradients_2/Mean_5_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
?
gradients_2/Mean_5_grad/ReshapeReshape1gradients_2/add_1_grad/tuple/control_dependency_1%gradients_2/Mean_5_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients_2/Mean_5_grad/ShapeShapepow_1*
_output_shapes
:*
out_type0*
T0
?
gradients_2/Mean_5_grad/TileTilegradients_2/Mean_5_grad/Reshapegradients_2/Mean_5_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:?????????
d
gradients_2/Mean_5_grad/Shape_1Shapepow_1*
_output_shapes
:*
out_type0*
T0
b
gradients_2/Mean_5_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_2/Mean_5_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
gradients_2/Mean_5_grad/ProdProdgradients_2/Mean_5_grad/Shape_1gradients_2/Mean_5_grad/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
i
gradients_2/Mean_5_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
?
gradients_2/Mean_5_grad/Prod_1Prodgradients_2/Mean_5_grad/Shape_2gradients_2/Mean_5_grad/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
c
!gradients_2/Mean_5_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
?
gradients_2/Mean_5_grad/MaximumMaximumgradients_2/Mean_5_grad/Prod_1!gradients_2/Mean_5_grad/Maximum/y*
T0*
_output_shapes
: 
?
 gradients_2/Mean_5_grad/floordivFloorDivgradients_2/Mean_5_grad/Prodgradients_2/Mean_5_grad/Maximum*
_output_shapes
: *
T0
?
gradients_2/Mean_5_grad/CastCast gradients_2/Mean_5_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
?
gradients_2/Mean_5_grad/truedivRealDivgradients_2/Mean_5_grad/Tilegradients_2/Mean_5_grad/Cast*#
_output_shapes
:?????????*
T0
_
gradients_2/pow_grad/ShapeShapesub_2*
out_type0*
_output_shapes
:*
T0
_
gradients_2/pow_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
?
*gradients_2/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_grad/Shapegradients_2/pow_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
u
gradients_2/pow_grad/mulMulgradients_2/Mean_4_grad/truedivpow/y*
T0*#
_output_shapes
:?????????
_
gradients_2/pow_grad/sub/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
c
gradients_2/pow_grad/subSubpow/ygradients_2/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_2/pow_grad/PowPowsub_2gradients_2/pow_grad/sub*
T0*#
_output_shapes
:?????????
?
gradients_2/pow_grad/mul_1Mulgradients_2/pow_grad/mulgradients_2/pow_grad/Pow*
T0*#
_output_shapes
:?????????
?
gradients_2/pow_grad/SumSumgradients_2/pow_grad/mul_1*gradients_2/pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
gradients_2/pow_grad/ReshapeReshapegradients_2/pow_grad/Sumgradients_2/pow_grad/Shape*#
_output_shapes
:?????????*
Tshape0*
T0
c
gradients_2/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
|
gradients_2/pow_grad/GreaterGreatersub_2gradients_2/pow_grad/Greater/y*
T0*#
_output_shapes
:?????????
i
$gradients_2/pow_grad/ones_like/ShapeShapesub_2*
out_type0*
T0*
_output_shapes
:
i
$gradients_2/pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
gradients_2/pow_grad/ones_likeFill$gradients_2/pow_grad/ones_like/Shape$gradients_2/pow_grad/ones_like/Const*#
_output_shapes
:?????????*

index_type0*
T0
?
gradients_2/pow_grad/SelectSelectgradients_2/pow_grad/Greatersub_2gradients_2/pow_grad/ones_like*#
_output_shapes
:?????????*
T0
j
gradients_2/pow_grad/LogLoggradients_2/pow_grad/Select*#
_output_shapes
:?????????*
T0
a
gradients_2/pow_grad/zeros_like	ZerosLikesub_2*
T0*#
_output_shapes
:?????????
?
gradients_2/pow_grad/Select_1Selectgradients_2/pow_grad/Greatergradients_2/pow_grad/Loggradients_2/pow_grad/zeros_like*
T0*#
_output_shapes
:?????????
u
gradients_2/pow_grad/mul_2Mulgradients_2/Mean_4_grad/truedivpow*
T0*#
_output_shapes
:?????????
?
gradients_2/pow_grad/mul_3Mulgradients_2/pow_grad/mul_2gradients_2/pow_grad/Select_1*#
_output_shapes
:?????????*
T0
?
gradients_2/pow_grad/Sum_1Sumgradients_2/pow_grad/mul_3,gradients_2/pow_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
gradients_2/pow_grad/Reshape_1Reshapegradients_2/pow_grad/Sum_1gradients_2/pow_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
m
%gradients_2/pow_grad/tuple/group_depsNoOp^gradients_2/pow_grad/Reshape^gradients_2/pow_grad/Reshape_1
?
-gradients_2/pow_grad/tuple/control_dependencyIdentitygradients_2/pow_grad/Reshape&^gradients_2/pow_grad/tuple/group_deps*
T0*#
_output_shapes
:?????????*/
_class%
#!loc:@gradients_2/pow_grad/Reshape
?
/gradients_2/pow_grad/tuple/control_dependency_1Identitygradients_2/pow_grad/Reshape_1&^gradients_2/pow_grad/tuple/group_deps*1
_class'
%#loc:@gradients_2/pow_grad/Reshape_1*
_output_shapes
: *
T0
a
gradients_2/pow_1_grad/ShapeShapesub_3*
_output_shapes
:*
T0*
out_type0
a
gradients_2/pow_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
,gradients_2/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_1_grad/Shapegradients_2/pow_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
y
gradients_2/pow_1_grad/mulMulgradients_2/Mean_5_grad/truedivpow_1/y*#
_output_shapes
:?????????*
T0
a
gradients_2/pow_1_grad/sub/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
i
gradients_2/pow_1_grad/subSubpow_1/ygradients_2/pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_2/pow_1_grad/PowPowsub_3gradients_2/pow_1_grad/sub*#
_output_shapes
:?????????*
T0
?
gradients_2/pow_1_grad/mul_1Mulgradients_2/pow_1_grad/mulgradients_2/pow_1_grad/Pow*
T0*#
_output_shapes
:?????????
?
gradients_2/pow_1_grad/SumSumgradients_2/pow_1_grad/mul_1,gradients_2/pow_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
?
gradients_2/pow_1_grad/ReshapeReshapegradients_2/pow_1_grad/Sumgradients_2/pow_1_grad/Shape*
Tshape0*
T0*#
_output_shapes
:?????????
e
 gradients_2/pow_1_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
gradients_2/pow_1_grad/GreaterGreatersub_3 gradients_2/pow_1_grad/Greater/y*
T0*#
_output_shapes
:?????????
k
&gradients_2/pow_1_grad/ones_like/ShapeShapesub_3*
out_type0*
_output_shapes
:*
T0
k
&gradients_2/pow_1_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
 gradients_2/pow_1_grad/ones_likeFill&gradients_2/pow_1_grad/ones_like/Shape&gradients_2/pow_1_grad/ones_like/Const*#
_output_shapes
:?????????*
T0*

index_type0
?
gradients_2/pow_1_grad/SelectSelectgradients_2/pow_1_grad/Greatersub_3 gradients_2/pow_1_grad/ones_like*#
_output_shapes
:?????????*
T0
n
gradients_2/pow_1_grad/LogLoggradients_2/pow_1_grad/Select*
T0*#
_output_shapes
:?????????
c
!gradients_2/pow_1_grad/zeros_like	ZerosLikesub_3*
T0*#
_output_shapes
:?????????
?
gradients_2/pow_1_grad/Select_1Selectgradients_2/pow_1_grad/Greatergradients_2/pow_1_grad/Log!gradients_2/pow_1_grad/zeros_like*#
_output_shapes
:?????????*
T0
y
gradients_2/pow_1_grad/mul_2Mulgradients_2/Mean_5_grad/truedivpow_1*
T0*#
_output_shapes
:?????????
?
gradients_2/pow_1_grad/mul_3Mulgradients_2/pow_1_grad/mul_2gradients_2/pow_1_grad/Select_1*#
_output_shapes
:?????????*
T0
?
gradients_2/pow_1_grad/Sum_1Sumgradients_2/pow_1_grad/mul_3.gradients_2/pow_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
 gradients_2/pow_1_grad/Reshape_1Reshapegradients_2/pow_1_grad/Sum_1gradients_2/pow_1_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
s
'gradients_2/pow_1_grad/tuple/group_depsNoOp^gradients_2/pow_1_grad/Reshape!^gradients_2/pow_1_grad/Reshape_1
?
/gradients_2/pow_1_grad/tuple/control_dependencyIdentitygradients_2/pow_1_grad/Reshape(^gradients_2/pow_1_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*1
_class'
%#loc:@gradients_2/pow_1_grad/Reshape
?
1gradients_2/pow_1_grad/tuple/control_dependency_1Identity gradients_2/pow_1_grad/Reshape_1(^gradients_2/pow_1_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients_2/pow_1_grad/Reshape_1
i
gradients_2/sub_2_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
h
gradients_2/sub_2_grad/Shape_1Shape
vf/Squeeze*
out_type0*
T0*
_output_shapes
:
?
,gradients_2/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_2_grad/Shapegradients_2/sub_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_2/sub_2_grad/SumSum-gradients_2/pow_grad/tuple/control_dependency,gradients_2/sub_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
gradients_2/sub_2_grad/ReshapeReshapegradients_2/sub_2_grad/Sumgradients_2/sub_2_grad/Shape*#
_output_shapes
:?????????*
Tshape0*
T0
?
gradients_2/sub_2_grad/Sum_1Sum-gradients_2/pow_grad/tuple/control_dependency.gradients_2/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
b
gradients_2/sub_2_grad/NegNeggradients_2/sub_2_grad/Sum_1*
T0*
_output_shapes
:
?
 gradients_2/sub_2_grad/Reshape_1Reshapegradients_2/sub_2_grad/Neggradients_2/sub_2_grad/Shape_1*
Tshape0*#
_output_shapes
:?????????*
T0
s
'gradients_2/sub_2_grad/tuple/group_depsNoOp^gradients_2/sub_2_grad/Reshape!^gradients_2/sub_2_grad/Reshape_1
?
/gradients_2/sub_2_grad/tuple/control_dependencyIdentitygradients_2/sub_2_grad/Reshape(^gradients_2/sub_2_grad/tuple/group_deps*
T0*#
_output_shapes
:?????????*1
_class'
%#loc:@gradients_2/sub_2_grad/Reshape
?
1gradients_2/sub_2_grad/tuple/control_dependency_1Identity gradients_2/sub_2_grad/Reshape_1(^gradients_2/sub_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_2/sub_2_grad/Reshape_1*#
_output_shapes
:?????????
i
gradients_2/sub_3_grad/ShapeShapePlaceholder_5*
_output_shapes
:*
T0*
out_type0
h
gradients_2/sub_3_grad/Shape_1Shape
vc/Squeeze*
T0*
_output_shapes
:*
out_type0
?
,gradients_2/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_3_grad/Shapegradients_2/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_2/sub_3_grad/SumSum/gradients_2/pow_1_grad/tuple/control_dependency,gradients_2/sub_3_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
gradients_2/sub_3_grad/ReshapeReshapegradients_2/sub_3_grad/Sumgradients_2/sub_3_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
?
gradients_2/sub_3_grad/Sum_1Sum/gradients_2/pow_1_grad/tuple/control_dependency.gradients_2/sub_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
b
gradients_2/sub_3_grad/NegNeggradients_2/sub_3_grad/Sum_1*
_output_shapes
:*
T0
?
 gradients_2/sub_3_grad/Reshape_1Reshapegradients_2/sub_3_grad/Neggradients_2/sub_3_grad/Shape_1*
T0*#
_output_shapes
:?????????*
Tshape0
s
'gradients_2/sub_3_grad/tuple/group_depsNoOp^gradients_2/sub_3_grad/Reshape!^gradients_2/sub_3_grad/Reshape_1
?
/gradients_2/sub_3_grad/tuple/control_dependencyIdentitygradients_2/sub_3_grad/Reshape(^gradients_2/sub_3_grad/tuple/group_deps*
T0*#
_output_shapes
:?????????*1
_class'
%#loc:@gradients_2/sub_3_grad/Reshape
?
1gradients_2/sub_3_grad/tuple/control_dependency_1Identity gradients_2/sub_3_grad/Reshape_1(^gradients_2/sub_3_grad/tuple/group_deps*
T0*#
_output_shapes
:?????????*3
_class)
'%loc:@gradients_2/sub_3_grad/Reshape_1
s
!gradients_2/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
?
#gradients_2/vf/Squeeze_grad/ReshapeReshape1gradients_2/sub_2_grad/tuple/control_dependency_1!gradients_2/vf/Squeeze_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
s
!gradients_2/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
?
#gradients_2/vc/Squeeze_grad/ReshapeReshape1gradients_2/sub_3_grad/tuple/control_dependency_1!gradients_2/vc/Squeeze_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
?
/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vf/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
?
4gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vf/Squeeze_grad/Reshape0^gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad
?
<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vf/Squeeze_grad/Reshape5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_2/vf/Squeeze_grad/Reshape*'
_output_shapes
:?????????
?
>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_2/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_2/vf/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
?
/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/vc/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
?
4gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_2/vc/Squeeze_grad/Reshape0^gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad
?
<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_2/vc/Squeeze_grad/Reshape5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*6
_class,
*(loc:@gradients_2/vc/Squeeze_grad/Reshape*'
_output_shapes
:?????????*
T0
?
>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_2/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_2/vc/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
?
)gradients_2/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*
T0*'
_output_shapes
:?????????@*
transpose_a( *
transpose_b(
?
+gradients_2/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes

:@*
T0
?
3gradients_2/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_2/MatMul_grad/MatMul,^gradients_2/vf/dense_2/MatMul_grad/MatMul_1
?
;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_2/MatMul_grad/MatMul4^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????@*<
_class2
0.loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul*
T0
?
=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_2/MatMul_grad/MatMul_14^gradients_2/vf/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_2/vf/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@*
T0
?
)gradients_2/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*
T0*'
_output_shapes
:?????????@*
transpose_a( *
transpose_b(
?
+gradients_2/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@*
T0*
transpose_a(
?
3gradients_2/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_2/MatMul_grad/MatMul,^gradients_2/vc/dense_2/MatMul_grad/MatMul_1
?
;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_2/MatMul_grad/MatMul4^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????@*<
_class2
0.loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul*
T0
?
=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_2/MatMul_grad/MatMul_14^gradients_2/vc/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_2/vc/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@*
T0
?
)gradients_2/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:?????????@*
T0
?
)gradients_2/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:?????????@*
T0
?
/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vf/dense_1/Tanh_grad/TanhGrad*
T0*
_output_shapes
:@*
data_formatNHWC
?
4gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vf/dense_1/Tanh_grad/TanhGrad
?
<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/Tanh_grad/TanhGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_2/vf/dense_1/Tanh_grad/TanhGrad*
T0*'
_output_shapes
:?????????@
?
>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vf/dense_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*B
_class8
64loc:@gradients_2/vf/dense_1/BiasAdd_grad/BiasAddGrad
?
/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/vc/dense_1/Tanh_grad/TanhGrad*
_output_shapes
:@*
T0*
data_formatNHWC
?
4gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_2/vc/dense_1/Tanh_grad/TanhGrad
?
<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/Tanh_grad/TanhGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????@*<
_class2
0.loc:@gradients_2/vc/dense_1/Tanh_grad/TanhGrad
?
>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_2/vc/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*B
_class8
64loc:@gradients_2/vc/dense_1/BiasAdd_grad/BiasAddGrad*
T0
?
)gradients_2/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:?????????@*
T0
?
+gradients_2/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
T0*
transpose_b( *
transpose_a(
?
3gradients_2/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vf/dense_1/MatMul_grad/MatMul,^gradients_2/vf/dense_1/MatMul_grad/MatMul_1
?
;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vf/dense_1/MatMul_grad/MatMul4^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul*'
_output_shapes
:?????????@
?
=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vf/dense_1/MatMul_grad/MatMul_14^gradients_2/vf/dense_1/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:@@*>
_class4
20loc:@gradients_2/vf/dense_1/MatMul_grad/MatMul_1
?
)gradients_2/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*'
_output_shapes
:?????????@*
transpose_a( *
T0*
transpose_b(
?
+gradients_2/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
T0*
transpose_b( *
transpose_a(
?
3gradients_2/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_2/vc/dense_1/MatMul_grad/MatMul,^gradients_2/vc/dense_1/MatMul_grad/MatMul_1
?
;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_2/vc/dense_1/MatMul_grad/MatMul4^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????@*<
_class2
0.loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul*
T0
?
=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_2/vc/dense_1/MatMul_grad/MatMul_14^gradients_2/vc/dense_1/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_2/vc/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@*
T0
?
'gradients_2/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency*'
_output_shapes
:?????????@*
T0
?
'gradients_2/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:?????????@
?
-gradients_2/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vf/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes
:@
?
2gradients_2/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vf/dense/Tanh_grad/TanhGrad
?
:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/Tanh_grad/TanhGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*:
_class0
.,loc:@gradients_2/vf/dense/Tanh_grad/TanhGrad*
T0*'
_output_shapes
:?????????@
?
<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vf/dense/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_2/vf/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
?
-gradients_2/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/vc/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:@*
T0
?
2gradients_2/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_2/vc/dense/Tanh_grad/TanhGrad
?
:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/Tanh_grad/TanhGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_2/vc/dense/Tanh_grad/TanhGrad*'
_output_shapes
:?????????@
?
<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_2/vc/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_2/vc/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
?
'gradients_2/vf/dense/MatMul_grad/MatMulMatMul:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*
transpose_b(*'
_output_shapes
:?????????<*
transpose_a( *
T0
?
)gradients_2/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes

:<@*
transpose_b( 
?
1gradients_2/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vf/dense/MatMul_grad/MatMul*^gradients_2/vf/dense/MatMul_grad/MatMul_1
?
9gradients_2/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vf/dense/MatMul_grad/MatMul2^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????<*
T0*:
_class0
.,loc:@gradients_2/vf/dense/MatMul_grad/MatMul
?
;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vf/dense/MatMul_grad/MatMul_12^gradients_2/vf/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:<@*<
_class2
0.loc:@gradients_2/vf/dense/MatMul_grad/MatMul_1*
T0
?
'gradients_2/vc/dense/MatMul_grad/MatMulMatMul:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*
transpose_a( *
transpose_b(*'
_output_shapes
:?????????<*
T0
?
)gradients_2/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:<@*
transpose_a(
?
1gradients_2/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_2/vc/dense/MatMul_grad/MatMul*^gradients_2/vc/dense/MatMul_grad/MatMul_1
?
9gradients_2/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_2/vc/dense/MatMul_grad/MatMul2^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????<*:
_class0
.,loc:@gradients_2/vc/dense/MatMul_grad/MatMul*
T0
?
;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_2/vc/dense/MatMul_grad/MatMul_12^gradients_2/vc/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:<@*
T0*<
_class2
0.loc:@gradients_2/vc/dense/MatMul_grad/MatMul_1
c
Reshape_52/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_52Reshape;gradients_2/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_52/shape*
T0*
Tshape0*
_output_shapes	
:?
c
Reshape_53/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?

Reshape_53Reshape<gradients_2/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_53/shape*
Tshape0*
T0*
_output_shapes
:@
c
Reshape_54/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?

Reshape_54Reshape=gradients_2/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_54/shape*
_output_shapes	
:? *
Tshape0*
T0
c
Reshape_55/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?

Reshape_55Reshape>gradients_2/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_55/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_56/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_56Reshape=gradients_2/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_56/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_57/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?

Reshape_57Reshape>gradients_2/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_57/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_58/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_58Reshape;gradients_2/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_58/shape*
Tshape0*
_output_shapes	
:?*
T0
c
Reshape_59/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_59Reshape<gradients_2/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_59/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_60/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_60Reshape=gradients_2/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_60/shape*
Tshape0*
T0*
_output_shapes	
:? 
c
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_61Reshape>gradients_2/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_61/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_62/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_62Reshape=gradients_2/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_62/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_63/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?

Reshape_63Reshape>gradients_2/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_63/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_4/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
concat_4ConcatV2
Reshape_52
Reshape_53
Reshape_54
Reshape_55
Reshape_56
Reshape_57
Reshape_58
Reshape_59
Reshape_60
Reshape_61
Reshape_62
Reshape_63concat_4/axis*
T0*
N*
_output_shapes	
:?*

Tidx0
k
PyFunc_4PyFuncconcat_4*
token
pyfunc_4*
Tin
2*
_output_shapes	
:?*
Tout
2
?
Const_10Const*
dtype0*E
value<B:"0   @      @   @         @      @   @      *
_output_shapes
:
S
split_4/split_dimConst*
value	B : *
_output_shapes
: *
dtype0
?
split_4SplitVPyFunc_4Const_10split_4/split_dim*
T0*
	num_split*`
_output_shapesN
L:?:@:? :@:@::?:@:? :@:@:*

Tlen0
a
Reshape_64/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   @   
g

Reshape_64Reshapesplit_4Reshape_64/shape*
T0*
Tshape0*
_output_shapes

:<@
Z
Reshape_65/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
e

Reshape_65Reshape	split_4:1Reshape_65/shape*
Tshape0*
T0*
_output_shapes
:@
a
Reshape_66/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   
i

Reshape_66Reshape	split_4:2Reshape_66/shape*
_output_shapes

:@@*
T0*
Tshape0
Z
Reshape_67/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_67Reshape	split_4:3Reshape_67/shape*
T0*
_output_shapes
:@*
Tshape0
a
Reshape_68/shapeConst*
valueB"@      *
_output_shapes
:*
dtype0
i

Reshape_68Reshape	split_4:4Reshape_68/shape*
_output_shapes

:@*
Tshape0*
T0
Z
Reshape_69/shapeConst*
_output_shapes
:*
valueB:*
dtype0
e

Reshape_69Reshape	split_4:5Reshape_69/shape*
_output_shapes
:*
T0*
Tshape0
a
Reshape_70/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   @   
i

Reshape_70Reshape	split_4:6Reshape_70/shape*
Tshape0*
_output_shapes

:<@*
T0
Z
Reshape_71/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_71Reshape	split_4:7Reshape_71/shape*
Tshape0*
_output_shapes
:@*
T0
a
Reshape_72/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
i

Reshape_72Reshape	split_4:8Reshape_72/shape*
_output_shapes

:@@*
T0*
Tshape0
Z
Reshape_73/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
e

Reshape_73Reshape	split_4:9Reshape_73/shape*
_output_shapes
:@*
Tshape0*
T0
a
Reshape_74/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0
j

Reshape_74Reshape
split_4:10Reshape_74/shape*
_output_shapes

:@*
T0*
Tshape0
Z
Reshape_75/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_75Reshape
split_4:11Reshape_75/shape*
_output_shapes
:*
Tshape0*
T0
?
beta1_power_2/initial_valueConst*
valueB
 *fff?*
_output_shapes
: *
dtype0* 
_class
loc:@vc/dense/bias
?
beta1_power_2
VariableV2* 
_class
loc:@vc/dense/bias*
shape: *
_output_shapes
: *
	container *
shared_name *
dtype0
?
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
p
beta1_power_2/readIdentitybeta1_power_2* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
?
beta2_power_2/initial_valueConst*
_output_shapes
: *
dtype0* 
_class
loc:@vc/dense/bias*
valueB
 *w??
?
beta2_power_2
VariableV2*
_output_shapes
: *
shape: * 
_class
loc:@vc/dense/bias*
shared_name *
dtype0*
	container 
?
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
T0
p
beta2_power_2/readIdentitybeta2_power_2*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
?
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"<   @   *
_output_shapes
:*"
_class
loc:@vf/dense/kernel
?
,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
?
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes

:<@*

index_type0*"
_class
loc:@vf/dense/kernel*
T0
?
vf/dense/kernel/Adam
VariableV2*
shape
:<@*
_output_shapes

:<@*"
_class
loc:@vf/dense/kernel*
shared_name *
dtype0*
	container 
?
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*
_output_shapes

:<@*
use_locking(*"
_class
loc:@vf/dense/kernel
?
vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*
T0*
_output_shapes

:<@*"
_class
loc:@vf/dense/kernel
?
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"<   @   *"
_class
loc:@vf/dense/kernel*
_output_shapes
:*
dtype0
?
.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes

:<@*

index_type0*"
_class
loc:@vf/dense/kernel
?
vf/dense/kernel/Adam_1
VariableV2*
_output_shapes

:<@*
dtype0*
shape
:<@*
	container *
shared_name *"
_class
loc:@vf/dense/kernel
?
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:<@*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
?
vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*"
_class
loc:@vf/dense/kernel*
_output_shapes

:<@*
T0
?
$vf/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
dtype0*
valueB@*    *
_output_shapes
:@
?
vf/dense/bias/Adam
VariableV2*
	container * 
_class
loc:@vf/dense/bias*
shared_name *
shape:@*
_output_shapes
:@*
dtype0
?
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes
:@*
T0*
validate_shape(
~
vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam* 
_class
loc:@vf/dense/bias*
_output_shapes
:@*
T0
?
&vf/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    * 
_class
loc:@vf/dense/bias*
dtype0
?
vf/dense/bias/Adam_1
VariableV2*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
shape:@*
shared_name *
	container *
dtype0
?
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
?
vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
T0
?
8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB"@   @   
?
.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0*$
_class
loc:@vf/dense_1/kernel
?
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@vf/dense_1/kernel*

index_type0*
_output_shapes

:@@
?
vf/dense_1/kernel/Adam
VariableV2*
shape
:@@*$
_class
loc:@vf/dense_1/kernel*
	container *
shared_name *
_output_shapes

:@@*
dtype0
?
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel
?
vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam*$
_class
loc:@vf/dense_1/kernel*
T0*
_output_shapes

:@@
?
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
:*
valueB"@   @   *
dtype0
?
0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *$
_class
loc:@vf/dense_1/kernel*
dtype0
?
*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const*$
_class
loc:@vf/dense_1/kernel*

index_type0*
T0*
_output_shapes

:@@
?
vf/dense_1/kernel/Adam_1
VariableV2*
	container *
shape
:@@*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
shared_name *
dtype0
?
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
T0
?
vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1*
T0*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel
?
&vf/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB@*    *"
_class
loc:@vf/dense_1/bias*
_output_shapes
:@
?
vf/dense_1/bias/Adam
VariableV2*
	container *
dtype0*
shape:@*
_output_shapes
:@*"
_class
loc:@vf/dense_1/bias*
shared_name 
?
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes
:@*
use_locking(
?
vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes
:@
?
(vf/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
_output_shapes
:@*
valueB@*    *
dtype0
?
vf/dense_1/bias/Adam_1
VariableV2*"
_class
loc:@vf/dense_1/bias*
dtype0*
	container *
shape:@*
shared_name *
_output_shapes
:@
?
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
_output_shapes
:@*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(
?
vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*
_output_shapes
:@*
T0*"
_class
loc:@vf/dense_1/bias
?
(vf/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@vf/dense_2/kernel*
valueB@*    *
_output_shapes

:@*
dtype0
?
vf/dense_2/kernel/Adam
VariableV2*$
_class
loc:@vf/dense_2/kernel*
dtype0*
_output_shapes

:@*
	container *
shape
:@*
shared_name 
?
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes

:@*
validate_shape(
?
vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*
_output_shapes

:@*$
_class
loc:@vf/dense_2/kernel*
T0
?
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:@*
valueB@*    *$
_class
loc:@vf/dense_2/kernel*
dtype0
?
vf/dense_2/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape
:@*$
_class
loc:@vf/dense_2/kernel*
	container *
_output_shapes

:@
?
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes

:@*
use_locking(
?
vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*
T0*
_output_shapes

:@*$
_class
loc:@vf/dense_2/kernel
?
&vf/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
dtype0*
valueB*    
?
vf/dense_2/bias/Adam
VariableV2*
	container *
shape:*"
_class
loc:@vf/dense_2/bias*
dtype0*
_output_shapes
:*
shared_name 
?
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
?
vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
?
(vf/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
valueB*    *
dtype0
?
vf/dense_2/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:*
_output_shapes
:*
shared_name *"
_class
loc:@vf/dense_2/bias
?
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(
?
vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0
?
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"<   @   *
_output_shapes
:*
dtype0*"
_class
loc:@vc/dense/kernel
?
,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *"
_class
loc:@vc/dense/kernel*
dtype0
?
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
T0*

index_type0
?
vc/dense/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@vc/dense/kernel*
	container *
dtype0*
shape
:<@*
_output_shapes

:<@
?
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
validate_shape(
?
vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
T0
?
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"<   @   *
_output_shapes
:*
dtype0*"
_class
loc:@vc/dense/kernel
?
.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *"
_class
loc:@vc/dense/kernel*
dtype0*
valueB
 *    
?
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*

index_type0*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
T0
?
vc/dense/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
shape
:<@*
	container 
?
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:<@*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(
?
vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*"
_class
loc:@vc/dense/kernel*
_output_shapes

:<@*
T0
?
$vc/dense/bias/Adam/Initializer/zerosConst*
dtype0*
valueB@*    *
_output_shapes
:@* 
_class
loc:@vc/dense/bias
?
vc/dense/bias/Adam
VariableV2*
_output_shapes
:@*
shared_name *
shape:@* 
_class
loc:@vc/dense/bias*
	container *
dtype0
?
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
validate_shape(
~
vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
:@
?
&vc/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
dtype0*
valueB@*    
?
vc/dense/bias/Adam_1
VariableV2*
shared_name *
_output_shapes
:@*
shape:@*
	container *
dtype0* 
_class
loc:@vc/dense/bias
?
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
?
vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
T0
?
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *
_output_shapes
:*
dtype0*$
_class
loc:@vc/dense_1/kernel
?
.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@vc/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const*

index_type0*
T0*$
_class
loc:@vc/dense_1/kernel*
_output_shapes

:@@
?
vc/dense_1/kernel/Adam
VariableV2*
shape
:@@*$
_class
loc:@vc/dense_1/kernel*
dtype0*
shared_name *
	container *
_output_shapes

:@@
?
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes

:@@*
T0*$
_class
loc:@vc/dense_1/kernel
?
vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam*$
_class
loc:@vc/dense_1/kernel*
_output_shapes

:@@*
T0
?
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"@   @   *
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel
?
0vc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const*

index_type0*$
_class
loc:@vc/dense_1/kernel*
T0*
_output_shapes

:@@
?
vc/dense_1/kernel/Adam_1
VariableV2*
shape
:@@*
_output_shapes

:@@*
dtype0*
	container *
shared_name *$
_class
loc:@vc/dense_1/kernel
?
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes

:@@*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
?
vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1*
T0*
_output_shapes

:@@*$
_class
loc:@vc/dense_1/kernel
?
&vc/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *
dtype0*"
_class
loc:@vc/dense_1/bias
?
vc/dense_1/bias/Adam
VariableV2*
dtype0*"
_class
loc:@vc/dense_1/bias*
shared_name *
shape:@*
_output_shapes
:@*
	container 
?
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0
?
vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*
T0*
_output_shapes
:@*"
_class
loc:@vc/dense_1/bias
?
(vc/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB@*    *"
_class
loc:@vc/dense_1/bias*
dtype0*
_output_shapes
:@
?
vc/dense_1/bias/Adam_1
VariableV2*
_output_shapes
:@*"
_class
loc:@vc/dense_1/bias*
dtype0*
	container *
shape:@*
shared_name 
?
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0
?
vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*"
_class
loc:@vc/dense_1/bias*
_output_shapes
:@*
T0
?
(vc/dense_2/kernel/Adam/Initializer/zerosConst*
valueB@*    *
dtype0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes

:@
?
vc/dense_2/kernel/Adam
VariableV2*
	container *$
_class
loc:@vc/dense_2/kernel*
shared_name *
shape
:@*
_output_shapes

:@*
dtype0
?
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*$
_class
loc:@vc/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
?
vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes

:@
?
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes

:@*
valueB@*    
?
vc/dense_2/kernel/Adam_1
VariableV2*
shape
:@*$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes

:@*
shared_name *
	container 
?
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0
?
vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes

:@
?
&vc/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@vc/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
?
vc/dense_2/bias/Adam
VariableV2*
shared_name *
shape:*
dtype0*
	container *"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
?
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
?
vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
?
(vc/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vc/dense_2/bias*
dtype0*
_output_shapes
:*
valueB*    
?
vc/dense_2/bias/Adam_1
VariableV2*
dtype0*
shape:*
	container *
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
shared_name 
?
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
?
vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:
Q
Adam_2/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
Q
Adam_2/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w??
S
Adam_2/epsilonConst*
dtype0*
valueB
 *w?+2*
_output_shapes
: 
?
'Adam_2/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_64*
use_locking( *"
_class
loc:@vf/dense/kernel*
use_nesterov( *
T0*
_output_shapes

:<@
?
%Adam_2/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_65* 
_class
loc:@vf/dense/bias*
T0*
use_locking( *
use_nesterov( *
_output_shapes
:@
?
)Adam_2/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_66*$
_class
loc:@vf/dense_1/kernel*
use_locking( *
_output_shapes

:@@*
T0*
use_nesterov( 
?
'Adam_2/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_67*
T0*
use_locking( *
_output_shapes
:@*"
_class
loc:@vf/dense_1/bias*
use_nesterov( 
?
)Adam_2/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_68*
use_nesterov( *
T0*
use_locking( *$
_class
loc:@vf/dense_2/kernel*
_output_shapes

:@
?
'Adam_2/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_69*
_output_shapes
:*
T0*"
_class
loc:@vf/dense_2/bias*
use_nesterov( *
use_locking( 
?
'Adam_2/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_70*"
_class
loc:@vc/dense/kernel*
use_locking( *
T0*
use_nesterov( *
_output_shapes

:<@
?
%Adam_2/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_71*
use_locking( *
use_nesterov( *
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
T0
?
)Adam_2/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_72*$
_class
loc:@vc/dense_1/kernel*
use_locking( *
use_nesterov( *
T0*
_output_shapes

:@@
?
'Adam_2/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_73*"
_class
loc:@vc/dense_1/bias*
use_nesterov( *
_output_shapes
:@*
T0*
use_locking( 
?
)Adam_2/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_74*
_output_shapes

:@*
use_locking( *
use_nesterov( *$
_class
loc:@vc/dense_2/kernel*
T0
?
'Adam_2/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readPolynomialDecay_2Adam_2/beta1Adam_2/beta2Adam_2/epsilon
Reshape_75*"
_class
loc:@vc/dense_2/bias*
use_nesterov( *
_output_shapes
:*
T0*
use_locking( 
?

Adam_2/mulMulbeta1_power_2/readAdam_2/beta1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias
?
Adam_2/AssignAssignbeta1_power_2
Adam_2/mul*
validate_shape(*
use_locking( * 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
?
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta2&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
?
Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
_output_shapes
: *
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking( *
T0
?
Adam_2NoOp^Adam_2/Assign^Adam_2/Assign_1&^Adam_2/update_vc/dense/bias/ApplyAdam(^Adam_2/update_vc/dense/kernel/ApplyAdam(^Adam_2/update_vc/dense_1/bias/ApplyAdam*^Adam_2/update_vc/dense_1/kernel/ApplyAdam(^Adam_2/update_vc/dense_2/bias/ApplyAdam*^Adam_2/update_vc/dense_2/kernel/ApplyAdam&^Adam_2/update_vf/dense/bias/ApplyAdam(^Adam_2/update_vf/dense/kernel/ApplyAdam(^Adam_2/update_vf/dense_1/bias/ApplyAdam*^Adam_2/update_vf/dense_1/kernel/ApplyAdam(^Adam_2/update_vf/dense_2/bias/ApplyAdam*^Adam_2/update_vf/dense_2/kernel/ApplyAdam
l
Reshape_76/shapeConst^Adam_2*
valueB:
?????????*
dtype0*
_output_shapes
:
q

Reshape_76Reshapevf/dense/kernel/readReshape_76/shape*
T0*
Tshape0*
_output_shapes	
:?
l
Reshape_77/shapeConst^Adam_2*
valueB:
?????????*
_output_shapes
:*
dtype0
n

Reshape_77Reshapevf/dense/bias/readReshape_77/shape*
_output_shapes
:@*
T0*
Tshape0
l
Reshape_78/shapeConst^Adam_2*
dtype0*
valueB:
?????????*
_output_shapes
:
s

Reshape_78Reshapevf/dense_1/kernel/readReshape_78/shape*
T0*
Tshape0*
_output_shapes	
:? 
l
Reshape_79/shapeConst^Adam_2*
valueB:
?????????*
_output_shapes
:*
dtype0
p

Reshape_79Reshapevf/dense_1/bias/readReshape_79/shape*
Tshape0*
T0*
_output_shapes
:@
l
Reshape_80/shapeConst^Adam_2*
valueB:
?????????*
dtype0*
_output_shapes
:
r

Reshape_80Reshapevf/dense_2/kernel/readReshape_80/shape*
_output_shapes
:@*
Tshape0*
T0
l
Reshape_81/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
?????????
p

Reshape_81Reshapevf/dense_2/bias/readReshape_81/shape*
_output_shapes
:*
T0*
Tshape0
l
Reshape_82/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
?????????
q

Reshape_82Reshapevc/dense/kernel/readReshape_82/shape*
Tshape0*
T0*
_output_shapes	
:?
l
Reshape_83/shapeConst^Adam_2*
dtype0*
valueB:
?????????*
_output_shapes
:
n

Reshape_83Reshapevc/dense/bias/readReshape_83/shape*
_output_shapes
:@*
Tshape0*
T0
l
Reshape_84/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:
?????????
s

Reshape_84Reshapevc/dense_1/kernel/readReshape_84/shape*
Tshape0*
_output_shapes	
:? *
T0
l
Reshape_85/shapeConst^Adam_2*
valueB:
?????????*
_output_shapes
:*
dtype0
p

Reshape_85Reshapevc/dense_1/bias/readReshape_85/shape*
_output_shapes
:@*
Tshape0*
T0
l
Reshape_86/shapeConst^Adam_2*
valueB:
?????????*
_output_shapes
:*
dtype0
r

Reshape_86Reshapevc/dense_2/kernel/readReshape_86/shape*
_output_shapes
:@*
T0*
Tshape0
l
Reshape_87/shapeConst^Adam_2*
_output_shapes
:*
valueB:
?????????*
dtype0
p

Reshape_87Reshapevc/dense_2/bias/readReshape_87/shape*
_output_shapes
:*
Tshape0*
T0
X
concat_5/axisConst^Adam_2*
dtype0*
_output_shapes
: *
value	B : 
?
concat_5ConcatV2
Reshape_76
Reshape_77
Reshape_78
Reshape_79
Reshape_80
Reshape_81
Reshape_82
Reshape_83
Reshape_84
Reshape_85
Reshape_86
Reshape_87concat_5/axis*
N*
_output_shapes	
:?*

Tidx0*
T0
h
PyFunc_5PyFuncconcat_5*
token
pyfunc_5*
_output_shapes
:*
Tout
2*
Tin
2
?
Const_11Const^Adam_2*E
value<B:"0   @      @   @         @      @   @      *
_output_shapes
:*
dtype0
\
split_5/split_dimConst^Adam_2*
dtype0*
_output_shapes
: *
value	B : 
?
split_5SplitVPyFunc_5Const_11split_5/split_dim*
	num_split*
T0*

Tlen0*D
_output_shapes2
0::::::::::::
j
Reshape_88/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB"<   @   
g

Reshape_88Reshapesplit_5Reshape_88/shape*
_output_shapes

:<@*
T0*
Tshape0
c
Reshape_89/shapeConst^Adam_2*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_89Reshape	split_5:1Reshape_89/shape*
T0*
_output_shapes
:@*
Tshape0
j
Reshape_90/shapeConst^Adam_2*
valueB"@   @   *
_output_shapes
:*
dtype0
i

Reshape_90Reshape	split_5:2Reshape_90/shape*
_output_shapes

:@@*
T0*
Tshape0
c
Reshape_91/shapeConst^Adam_2*
valueB:@*
_output_shapes
:*
dtype0
e

Reshape_91Reshape	split_5:3Reshape_91/shape*
Tshape0*
T0*
_output_shapes
:@
j
Reshape_92/shapeConst^Adam_2*
dtype0*
valueB"@      *
_output_shapes
:
i

Reshape_92Reshape	split_5:4Reshape_92/shape*
T0*
_output_shapes

:@*
Tshape0
c
Reshape_93/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_93Reshape	split_5:5Reshape_93/shape*
_output_shapes
:*
T0*
Tshape0
j
Reshape_94/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB"<   @   
i

Reshape_94Reshape	split_5:6Reshape_94/shape*
_output_shapes

:<@*
T0*
Tshape0
c
Reshape_95/shapeConst^Adam_2*
_output_shapes
:*
dtype0*
valueB:@
e

Reshape_95Reshape	split_5:7Reshape_95/shape*
T0*
_output_shapes
:@*
Tshape0
j
Reshape_96/shapeConst^Adam_2*
dtype0*
valueB"@   @   *
_output_shapes
:
i

Reshape_96Reshape	split_5:8Reshape_96/shape*
T0*
Tshape0*
_output_shapes

:@@
c
Reshape_97/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:@
e

Reshape_97Reshape	split_5:9Reshape_97/shape*
T0*
Tshape0*
_output_shapes
:@
j
Reshape_98/shapeConst^Adam_2*
valueB"@      *
_output_shapes
:*
dtype0
j

Reshape_98Reshape
split_5:10Reshape_98/shape*
T0*
Tshape0*
_output_shapes

:@
c
Reshape_99/shapeConst^Adam_2*
dtype0*
_output_shapes
:*
valueB:
f

Reshape_99Reshape
split_5:11Reshape_99/shape*
T0*
_output_shapes
:*
Tshape0
?
	Assign_13Assignvf/dense/kernel
Reshape_88*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes

:<@*
use_locking(*
validate_shape(
?
	Assign_14Assignvf/dense/bias
Reshape_89*
use_locking(*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0
?
	Assign_15Assignvf/dense_1/kernel
Reshape_90*
use_locking(*
validate_shape(*
_output_shapes

:@@*
T0*$
_class
loc:@vf/dense_1/kernel
?
	Assign_16Assignvf/dense_1/bias
Reshape_91*
validate_shape(*
use_locking(*
_output_shapes
:@*"
_class
loc:@vf/dense_1/bias*
T0
?
	Assign_17Assignvf/dense_2/kernel
Reshape_92*
use_locking(*
_output_shapes

:@*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(
?
	Assign_18Assignvf/dense_2/bias
Reshape_93*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias
?
	Assign_19Assignvc/dense/kernel
Reshape_94*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes

:<@*
validate_shape(*
T0
?
	Assign_20Assignvc/dense/bias
Reshape_95* 
_class
loc:@vc/dense/bias*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
?
	Assign_21Assignvc/dense_1/kernel
Reshape_96*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
?
	Assign_22Assignvc/dense_1/bias
Reshape_97*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
?
	Assign_23Assignvc/dense_2/kernel
Reshape_98*
_output_shapes

:@*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(*
T0
?
	Assign_24Assignvc/dense_2/bias
Reshape_99*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
?
group_deps_4NoOp^Adam_2
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
,
group_deps_5NoOp^Adam_2^group_deps_4
?
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^beta1_power/Assign^beta1_power_1/Assign^beta1_power_2/Assign^beta2_power/Assign^beta2_power_1/Assign^beta2_power_2/Assign^lam/dense/bias/Adam/Assign^lam/dense/bias/Adam_1/Assign^lam/dense/bias/Assign^lam/dense/kernel/Adam/Assign^lam/dense/kernel/Adam_1/Assign^lam/dense/kernel/Assign^lam/dense_1/bias/Adam/Assign^lam/dense_1/bias/Adam_1/Assign^lam/dense_1/bias/Assign^lam/dense_1/kernel/Adam/Assign!^lam/dense_1/kernel/Adam_1/Assign^lam/dense_1/kernel/Assign^lam/dense_2/bias/Adam/Assign^lam/dense_2/bias/Adam_1/Assign^lam/dense_2/bias/Assign^lam/dense_2/kernel/Adam/Assign!^lam/dense_2/kernel/Adam_1/Assign^lam/dense_2/kernel/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^pi/log_std/Adam/Assign^pi/log_std/Adam_1/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
d
Reshape_100/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
s
Reshape_100Reshapepi/dense/kernel/readReshape_100/shape*
Tshape0*
_output_shapes	
:?*
T0
d
Reshape_101/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
p
Reshape_101Reshapepi/dense/bias/readReshape_101/shape*
T0*
_output_shapes
:@*
Tshape0
d
Reshape_102/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
u
Reshape_102Reshapepi/dense_1/kernel/readReshape_102/shape*
_output_shapes	
:? *
T0*
Tshape0
d
Reshape_103/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
r
Reshape_103Reshapepi/dense_1/bias/readReshape_103/shape*
Tshape0*
T0*
_output_shapes
:@
d
Reshape_104/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
u
Reshape_104Reshapepi/dense_2/kernel/readReshape_104/shape*
Tshape0*
_output_shapes	
:?*
T0
d
Reshape_105/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
r
Reshape_105Reshapepi/dense_2/bias/readReshape_105/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_106/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
m
Reshape_106Reshapepi/log_std/readReshape_106/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_107/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
s
Reshape_107Reshapevf/dense/kernel/readReshape_107/shape*
T0*
_output_shapes	
:?*
Tshape0
d
Reshape_108/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
p
Reshape_108Reshapevf/dense/bias/readReshape_108/shape*
Tshape0*
T0*
_output_shapes
:@
d
Reshape_109/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
u
Reshape_109Reshapevf/dense_1/kernel/readReshape_109/shape*
T0*
Tshape0*
_output_shapes	
:? 
d
Reshape_110/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
r
Reshape_110Reshapevf/dense_1/bias/readReshape_110/shape*
_output_shapes
:@*
T0*
Tshape0
d
Reshape_111/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
t
Reshape_111Reshapevf/dense_2/kernel/readReshape_111/shape*
_output_shapes
:@*
T0*
Tshape0
d
Reshape_112/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
r
Reshape_112Reshapevf/dense_2/bias/readReshape_112/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_113/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
s
Reshape_113Reshapevc/dense/kernel/readReshape_113/shape*
Tshape0*
_output_shapes	
:?*
T0
d
Reshape_114/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
p
Reshape_114Reshapevc/dense/bias/readReshape_114/shape*
T0*
_output_shapes
:@*
Tshape0
d
Reshape_115/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
u
Reshape_115Reshapevc/dense_1/kernel/readReshape_115/shape*
Tshape0*
_output_shapes	
:? *
T0
d
Reshape_116/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
r
Reshape_116Reshapevc/dense_1/bias/readReshape_116/shape*
_output_shapes
:@*
T0*
Tshape0
d
Reshape_117/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
t
Reshape_117Reshapevc/dense_2/kernel/readReshape_117/shape*
T0*
Tshape0*
_output_shapes
:@
d
Reshape_118/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
r
Reshape_118Reshapevc/dense_2/bias/readReshape_118/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_119/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
t
Reshape_119Reshapelam/dense/kernel/readReshape_119/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_120/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
q
Reshape_120Reshapelam/dense/bias/readReshape_120/shape*
Tshape0*
T0*
_output_shapes
:@
d
Reshape_121/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
v
Reshape_121Reshapelam/dense_1/kernel/readReshape_121/shape*
T0*
Tshape0*
_output_shapes	
:? 
d
Reshape_122/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
s
Reshape_122Reshapelam/dense_1/bias/readReshape_122/shape*
Tshape0*
_output_shapes
:@*
T0
d
Reshape_123/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
u
Reshape_123Reshapelam/dense_2/kernel/readReshape_123/shape*
T0*
Tshape0*
_output_shapes
:@
d
Reshape_124/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
s
Reshape_124Reshapelam/dense_2/bias/readReshape_124/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_125/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
k
Reshape_125ReshapeVariable/readReshape_125/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_126/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
n
Reshape_126Reshapebeta1_power/readReshape_126/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_127/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
n
Reshape_127Reshapebeta2_power/readReshape_127/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_128/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
x
Reshape_128Reshapepi/dense/kernel/Adam/readReshape_128/shape*
T0*
Tshape0*
_output_shapes	
:?
d
Reshape_129/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
z
Reshape_129Reshapepi/dense/kernel/Adam_1/readReshape_129/shape*
T0*
_output_shapes	
:?*
Tshape0
d
Reshape_130/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
u
Reshape_130Reshapepi/dense/bias/Adam/readReshape_130/shape*
Tshape0*
_output_shapes
:@*
T0
d
Reshape_131/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
w
Reshape_131Reshapepi/dense/bias/Adam_1/readReshape_131/shape*
Tshape0*
_output_shapes
:@*
T0
d
Reshape_132/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
z
Reshape_132Reshapepi/dense_1/kernel/Adam/readReshape_132/shape*
T0*
_output_shapes	
:? *
Tshape0
d
Reshape_133/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
|
Reshape_133Reshapepi/dense_1/kernel/Adam_1/readReshape_133/shape*
_output_shapes	
:? *
T0*
Tshape0
d
Reshape_134/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
w
Reshape_134Reshapepi/dense_1/bias/Adam/readReshape_134/shape*
T0*
_output_shapes
:@*
Tshape0
d
Reshape_135/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
y
Reshape_135Reshapepi/dense_1/bias/Adam_1/readReshape_135/shape*
Tshape0*
T0*
_output_shapes
:@
d
Reshape_136/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
z
Reshape_136Reshapepi/dense_2/kernel/Adam/readReshape_136/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_137/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
|
Reshape_137Reshapepi/dense_2/kernel/Adam_1/readReshape_137/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_138/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
w
Reshape_138Reshapepi/dense_2/bias/Adam/readReshape_138/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_139/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
y
Reshape_139Reshapepi/dense_2/bias/Adam_1/readReshape_139/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_140/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
r
Reshape_140Reshapepi/log_std/Adam/readReshape_140/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_141/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
t
Reshape_141Reshapepi/log_std/Adam_1/readReshape_141/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_142/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
m
Reshape_142ReshapeVariable_1/readReshape_142/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_143/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
p
Reshape_143Reshapebeta1_power_1/readReshape_143/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_144/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
p
Reshape_144Reshapebeta2_power_1/readReshape_144/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_145/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
y
Reshape_145Reshapelam/dense/kernel/Adam/readReshape_145/shape*
Tshape0*
_output_shapes	
:?*
T0
d
Reshape_146/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
{
Reshape_146Reshapelam/dense/kernel/Adam_1/readReshape_146/shape*
T0*
Tshape0*
_output_shapes	
:?
d
Reshape_147/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
v
Reshape_147Reshapelam/dense/bias/Adam/readReshape_147/shape*
Tshape0*
T0*
_output_shapes
:@
d
Reshape_148/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
x
Reshape_148Reshapelam/dense/bias/Adam_1/readReshape_148/shape*
T0*
_output_shapes
:@*
Tshape0
d
Reshape_149/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
{
Reshape_149Reshapelam/dense_1/kernel/Adam/readReshape_149/shape*
_output_shapes	
:? *
T0*
Tshape0
d
Reshape_150/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
}
Reshape_150Reshapelam/dense_1/kernel/Adam_1/readReshape_150/shape*
T0*
_output_shapes	
:? *
Tshape0
d
Reshape_151/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
x
Reshape_151Reshapelam/dense_1/bias/Adam/readReshape_151/shape*
Tshape0*
T0*
_output_shapes
:@
d
Reshape_152/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
z
Reshape_152Reshapelam/dense_1/bias/Adam_1/readReshape_152/shape*
Tshape0*
_output_shapes
:@*
T0
d
Reshape_153/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
z
Reshape_153Reshapelam/dense_2/kernel/Adam/readReshape_153/shape*
Tshape0*
_output_shapes
:@*
T0
d
Reshape_154/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
|
Reshape_154Reshapelam/dense_2/kernel/Adam_1/readReshape_154/shape*
T0*
_output_shapes
:@*
Tshape0
d
Reshape_155/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
x
Reshape_155Reshapelam/dense_2/bias/Adam/readReshape_155/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_156/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
z
Reshape_156Reshapelam/dense_2/bias/Adam_1/readReshape_156/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_157/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
m
Reshape_157ReshapeVariable_2/readReshape_157/shape*
Tshape0*
T0*
_output_shapes
:
d
Reshape_158/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
p
Reshape_158Reshapebeta1_power_2/readReshape_158/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_159/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
p
Reshape_159Reshapebeta2_power_2/readReshape_159/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_160/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
x
Reshape_160Reshapevf/dense/kernel/Adam/readReshape_160/shape*
_output_shapes	
:?*
T0*
Tshape0
d
Reshape_161/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
z
Reshape_161Reshapevf/dense/kernel/Adam_1/readReshape_161/shape*
T0*
Tshape0*
_output_shapes	
:?
d
Reshape_162/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
u
Reshape_162Reshapevf/dense/bias/Adam/readReshape_162/shape*
_output_shapes
:@*
Tshape0*
T0
d
Reshape_163/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
w
Reshape_163Reshapevf/dense/bias/Adam_1/readReshape_163/shape*
Tshape0*
T0*
_output_shapes
:@
d
Reshape_164/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
z
Reshape_164Reshapevf/dense_1/kernel/Adam/readReshape_164/shape*
Tshape0*
T0*
_output_shapes	
:? 
d
Reshape_165/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
|
Reshape_165Reshapevf/dense_1/kernel/Adam_1/readReshape_165/shape*
Tshape0*
_output_shapes	
:? *
T0
d
Reshape_166/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
w
Reshape_166Reshapevf/dense_1/bias/Adam/readReshape_166/shape*
_output_shapes
:@*
Tshape0*
T0
d
Reshape_167/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
y
Reshape_167Reshapevf/dense_1/bias/Adam_1/readReshape_167/shape*
_output_shapes
:@*
T0*
Tshape0
d
Reshape_168/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
y
Reshape_168Reshapevf/dense_2/kernel/Adam/readReshape_168/shape*
_output_shapes
:@*
Tshape0*
T0
d
Reshape_169/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
{
Reshape_169Reshapevf/dense_2/kernel/Adam_1/readReshape_169/shape*
_output_shapes
:@*
Tshape0*
T0
d
Reshape_170/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
w
Reshape_170Reshapevf/dense_2/bias/Adam/readReshape_170/shape*
_output_shapes
:*
T0*
Tshape0
d
Reshape_171/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
y
Reshape_171Reshapevf/dense_2/bias/Adam_1/readReshape_171/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_172/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
x
Reshape_172Reshapevc/dense/kernel/Adam/readReshape_172/shape*
_output_shapes	
:?*
Tshape0*
T0
d
Reshape_173/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
z
Reshape_173Reshapevc/dense/kernel/Adam_1/readReshape_173/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_174/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
u
Reshape_174Reshapevc/dense/bias/Adam/readReshape_174/shape*
_output_shapes
:@*
Tshape0*
T0
d
Reshape_175/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
w
Reshape_175Reshapevc/dense/bias/Adam_1/readReshape_175/shape*
Tshape0*
_output_shapes
:@*
T0
d
Reshape_176/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
z
Reshape_176Reshapevc/dense_1/kernel/Adam/readReshape_176/shape*
T0*
_output_shapes	
:? *
Tshape0
d
Reshape_177/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
|
Reshape_177Reshapevc/dense_1/kernel/Adam_1/readReshape_177/shape*
Tshape0*
T0*
_output_shapes	
:? 
d
Reshape_178/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
w
Reshape_178Reshapevc/dense_1/bias/Adam/readReshape_178/shape*
T0*
_output_shapes
:@*
Tshape0
d
Reshape_179/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
y
Reshape_179Reshapevc/dense_1/bias/Adam_1/readReshape_179/shape*
Tshape0*
_output_shapes
:@*
T0
d
Reshape_180/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
y
Reshape_180Reshapevc/dense_2/kernel/Adam/readReshape_180/shape*
T0*
Tshape0*
_output_shapes
:@
d
Reshape_181/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
{
Reshape_181Reshapevc/dense_2/kernel/Adam_1/readReshape_181/shape*
T0*
Tshape0*
_output_shapes
:@
d
Reshape_182/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
w
Reshape_182Reshapevc/dense_2/bias/Adam/readReshape_182/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_183/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
y
Reshape_183Reshapevc/dense_2/bias/Adam_1/readReshape_183/shape*
Tshape0*
_output_shapes
:*
T0
O
concat_6/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?	
concat_6ConcatV2Reshape_100Reshape_101Reshape_102Reshape_103Reshape_104Reshape_105Reshape_106Reshape_107Reshape_108Reshape_109Reshape_110Reshape_111Reshape_112Reshape_113Reshape_114Reshape_115Reshape_116Reshape_117Reshape_118Reshape_119Reshape_120Reshape_121Reshape_122Reshape_123Reshape_124Reshape_125Reshape_126Reshape_127Reshape_128Reshape_129Reshape_130Reshape_131Reshape_132Reshape_133Reshape_134Reshape_135Reshape_136Reshape_137Reshape_138Reshape_139Reshape_140Reshape_141Reshape_142Reshape_143Reshape_144Reshape_145Reshape_146Reshape_147Reshape_148Reshape_149Reshape_150Reshape_151Reshape_152Reshape_153Reshape_154Reshape_155Reshape_156Reshape_157Reshape_158Reshape_159Reshape_160Reshape_161Reshape_162Reshape_163Reshape_164Reshape_165Reshape_166Reshape_167Reshape_168Reshape_169Reshape_170Reshape_171Reshape_172Reshape_173Reshape_174Reshape_175Reshape_176Reshape_177Reshape_178Reshape_179Reshape_180Reshape_181Reshape_182Reshape_183concat_6/axis*
T0*
NT*

Tidx0*
_output_shapes

:??
h
PyFunc_6PyFuncconcat_6*
token
pyfunc_6*
_output_shapes
:*
Tin
2*
Tout
2
?
Const_12Const*
_output_shapes
:T*
dtype0*?
value?B?T"?   @      @   ?            @      @   @         @      @   @         @      @   @                     @   @         @   @   ?   ?                              @   @         @   @   @   @                        @   @         @   @   @   @               @   @         @   @   @   @         
S
split_6/split_dimConst*
dtype0*
_output_shapes
: *
value	B : 
?
split_6SplitVPyFunc_6Const_12split_6/split_dim*
	num_splitT*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*

Tlen0*
T0
b
Reshape_184/shapeConst*
_output_shapes
:*
valueB"<   @   *
dtype0
i
Reshape_184Reshapesplit_6Reshape_184/shape*
_output_shapes

:<@*
T0*
Tshape0
[
Reshape_185/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
g
Reshape_185Reshape	split_6:1Reshape_185/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_186/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
k
Reshape_186Reshape	split_6:2Reshape_186/shape*
T0*
Tshape0*
_output_shapes

:@@
[
Reshape_187/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
g
Reshape_187Reshape	split_6:3Reshape_187/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_188/shapeConst*
valueB"@      *
_output_shapes
:*
dtype0
k
Reshape_188Reshape	split_6:4Reshape_188/shape*
_output_shapes

:@*
Tshape0*
T0
[
Reshape_189/shapeConst*
valueB:*
dtype0*
_output_shapes
:
g
Reshape_189Reshape	split_6:5Reshape_189/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_190/shapeConst*
_output_shapes
:*
valueB:*
dtype0
g
Reshape_190Reshape	split_6:6Reshape_190/shape*
T0*
Tshape0*
_output_shapes
:
b
Reshape_191/shapeConst*
_output_shapes
:*
valueB"<   @   *
dtype0
k
Reshape_191Reshape	split_6:7Reshape_191/shape*
Tshape0*
T0*
_output_shapes

:<@
[
Reshape_192/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
g
Reshape_192Reshape	split_6:8Reshape_192/shape*
Tshape0*
T0*
_output_shapes
:@
b
Reshape_193/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
k
Reshape_193Reshape	split_6:9Reshape_193/shape*
T0*
Tshape0*
_output_shapes

:@@
[
Reshape_194/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_194Reshape
split_6:10Reshape_194/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_195/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_195Reshape
split_6:11Reshape_195/shape*
Tshape0*
T0*
_output_shapes

:@
[
Reshape_196/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_196Reshape
split_6:12Reshape_196/shape*
Tshape0*
T0*
_output_shapes
:
b
Reshape_197/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   @   
l
Reshape_197Reshape
split_6:13Reshape_197/shape*
T0*
Tshape0*
_output_shapes

:<@
[
Reshape_198/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_198Reshape
split_6:14Reshape_198/shape*
Tshape0*
_output_shapes
:@*
T0
b
Reshape_199/shapeConst*
valueB"@   @   *
_output_shapes
:*
dtype0
l
Reshape_199Reshape
split_6:15Reshape_199/shape*
Tshape0*
T0*
_output_shapes

:@@
[
Reshape_200/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_200Reshape
split_6:16Reshape_200/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_201/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      
l
Reshape_201Reshape
split_6:17Reshape_201/shape*
T0*
Tshape0*
_output_shapes

:@
[
Reshape_202/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_202Reshape
split_6:18Reshape_202/shape*
_output_shapes
:*
T0*
Tshape0
b
Reshape_203/shapeConst*
valueB"<   @   *
dtype0*
_output_shapes
:
l
Reshape_203Reshape
split_6:19Reshape_203/shape*
Tshape0*
_output_shapes

:<@*
T0
[
Reshape_204/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
h
Reshape_204Reshape
split_6:20Reshape_204/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_205/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   
l
Reshape_205Reshape
split_6:21Reshape_205/shape*
T0*
Tshape0*
_output_shapes

:@@
[
Reshape_206/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
h
Reshape_206Reshape
split_6:22Reshape_206/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_207/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0
l
Reshape_207Reshape
split_6:23Reshape_207/shape*
T0*
Tshape0*
_output_shapes

:@
[
Reshape_208/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_208Reshape
split_6:24Reshape_208/shape*
_output_shapes
:*
Tshape0*
T0
T
Reshape_209/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_209Reshape
split_6:25Reshape_209/shape*
_output_shapes
: *
Tshape0*
T0
T
Reshape_210/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_210Reshape
split_6:26Reshape_210/shape*
Tshape0*
T0*
_output_shapes
: 
T
Reshape_211/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_211Reshape
split_6:27Reshape_211/shape*
T0*
_output_shapes
: *
Tshape0
b
Reshape_212/shapeConst*
dtype0*
valueB"<   @   *
_output_shapes
:
l
Reshape_212Reshape
split_6:28Reshape_212/shape*
Tshape0*
_output_shapes

:<@*
T0
b
Reshape_213/shapeConst*
valueB"<   @   *
dtype0*
_output_shapes
:
l
Reshape_213Reshape
split_6:29Reshape_213/shape*
T0*
_output_shapes

:<@*
Tshape0
[
Reshape_214/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
h
Reshape_214Reshape
split_6:30Reshape_214/shape*
T0*
Tshape0*
_output_shapes
:@
[
Reshape_215/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
h
Reshape_215Reshape
split_6:31Reshape_215/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_216/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
l
Reshape_216Reshape
split_6:32Reshape_216/shape*
Tshape0*
T0*
_output_shapes

:@@
b
Reshape_217/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   
l
Reshape_217Reshape
split_6:33Reshape_217/shape*
T0*
_output_shapes

:@@*
Tshape0
[
Reshape_218/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_218Reshape
split_6:34Reshape_218/shape*
T0*
Tshape0*
_output_shapes
:@
[
Reshape_219/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_219Reshape
split_6:35Reshape_219/shape*
Tshape0*
_output_shapes
:@*
T0
b
Reshape_220/shapeConst*
dtype0*
valueB"@      *
_output_shapes
:
l
Reshape_220Reshape
split_6:36Reshape_220/shape*
_output_shapes

:@*
Tshape0*
T0
b
Reshape_221/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      
l
Reshape_221Reshape
split_6:37Reshape_221/shape*
Tshape0*
_output_shapes

:@*
T0
[
Reshape_222/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_222Reshape
split_6:38Reshape_222/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_223/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_223Reshape
split_6:39Reshape_223/shape*
Tshape0*
T0*
_output_shapes
:
[
Reshape_224/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_224Reshape
split_6:40Reshape_224/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_225/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_225Reshape
split_6:41Reshape_225/shape*
Tshape0*
T0*
_output_shapes
:
T
Reshape_226/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_226Reshape
split_6:42Reshape_226/shape*
Tshape0*
T0*
_output_shapes
: 
T
Reshape_227/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_227Reshape
split_6:43Reshape_227/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_228/shapeConst*
_output_shapes
: *
valueB *
dtype0
d
Reshape_228Reshape
split_6:44Reshape_228/shape*
Tshape0*
_output_shapes
: *
T0
b
Reshape_229/shapeConst*
_output_shapes
:*
valueB"<   @   *
dtype0
l
Reshape_229Reshape
split_6:45Reshape_229/shape*
T0*
Tshape0*
_output_shapes

:<@
b
Reshape_230/shapeConst*
valueB"<   @   *
dtype0*
_output_shapes
:
l
Reshape_230Reshape
split_6:46Reshape_230/shape*
T0*
Tshape0*
_output_shapes

:<@
[
Reshape_231/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
h
Reshape_231Reshape
split_6:47Reshape_231/shape*
Tshape0*
T0*
_output_shapes
:@
[
Reshape_232/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_232Reshape
split_6:48Reshape_232/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_233/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
l
Reshape_233Reshape
split_6:49Reshape_233/shape*
T0*
_output_shapes

:@@*
Tshape0
b
Reshape_234/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   
l
Reshape_234Reshape
split_6:50Reshape_234/shape*
_output_shapes

:@@*
T0*
Tshape0
[
Reshape_235/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_235Reshape
split_6:51Reshape_235/shape*
Tshape0*
T0*
_output_shapes
:@
[
Reshape_236/shapeConst*
dtype0*
_output_shapes
:*
valueB:@
h
Reshape_236Reshape
split_6:52Reshape_236/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_237/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0
l
Reshape_237Reshape
split_6:53Reshape_237/shape*
Tshape0*
_output_shapes

:@*
T0
b
Reshape_238/shapeConst*
valueB"@      *
_output_shapes
:*
dtype0
l
Reshape_238Reshape
split_6:54Reshape_238/shape*
Tshape0*
T0*
_output_shapes

:@
[
Reshape_239/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_239Reshape
split_6:55Reshape_239/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_240/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_240Reshape
split_6:56Reshape_240/shape*
_output_shapes
:*
T0*
Tshape0
T
Reshape_241/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_241Reshape
split_6:57Reshape_241/shape*
Tshape0*
T0*
_output_shapes
: 
T
Reshape_242/shapeConst*
dtype0*
valueB *
_output_shapes
: 
d
Reshape_242Reshape
split_6:58Reshape_242/shape*
_output_shapes
: *
Tshape0*
T0
T
Reshape_243/shapeConst*
valueB *
_output_shapes
: *
dtype0
d
Reshape_243Reshape
split_6:59Reshape_243/shape*
Tshape0*
_output_shapes
: *
T0
b
Reshape_244/shapeConst*
valueB"<   @   *
dtype0*
_output_shapes
:
l
Reshape_244Reshape
split_6:60Reshape_244/shape*
Tshape0*
_output_shapes

:<@*
T0
b
Reshape_245/shapeConst*
dtype0*
_output_shapes
:*
valueB"<   @   
l
Reshape_245Reshape
split_6:61Reshape_245/shape*
_output_shapes

:<@*
T0*
Tshape0
[
Reshape_246/shapeConst*
_output_shapes
:*
dtype0*
valueB:@
h
Reshape_246Reshape
split_6:62Reshape_246/shape*
Tshape0*
T0*
_output_shapes
:@
[
Reshape_247/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
h
Reshape_247Reshape
split_6:63Reshape_247/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_248/shapeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
l
Reshape_248Reshape
split_6:64Reshape_248/shape*
_output_shapes

:@@*
Tshape0*
T0
b
Reshape_249/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   @   
l
Reshape_249Reshape
split_6:65Reshape_249/shape*
_output_shapes

:@@*
T0*
Tshape0
[
Reshape_250/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
h
Reshape_250Reshape
split_6:66Reshape_250/shape*
_output_shapes
:@*
T0*
Tshape0
[
Reshape_251/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
h
Reshape_251Reshape
split_6:67Reshape_251/shape*
Tshape0*
T0*
_output_shapes
:@
b
Reshape_252/shapeConst*
valueB"@      *
_output_shapes
:*
dtype0
l
Reshape_252Reshape
split_6:68Reshape_252/shape*
T0*
_output_shapes

:@*
Tshape0
b
Reshape_253/shapeConst*
dtype0*
_output_shapes
:*
valueB"@      
l
Reshape_253Reshape
split_6:69Reshape_253/shape*
Tshape0*
T0*
_output_shapes

:@
[
Reshape_254/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_254Reshape
split_6:70Reshape_254/shape*
_output_shapes
:*
T0*
Tshape0
[
Reshape_255/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_255Reshape
split_6:71Reshape_255/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_256/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   @   
l
Reshape_256Reshape
split_6:72Reshape_256/shape*
Tshape0*
T0*
_output_shapes

:<@
b
Reshape_257/shapeConst*
_output_shapes
:*
valueB"<   @   *
dtype0
l
Reshape_257Reshape
split_6:73Reshape_257/shape*
T0*
Tshape0*
_output_shapes

:<@
[
Reshape_258/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_258Reshape
split_6:74Reshape_258/shape*
_output_shapes
:@*
Tshape0*
T0
[
Reshape_259/shapeConst*
valueB:@*
_output_shapes
:*
dtype0
h
Reshape_259Reshape
split_6:75Reshape_259/shape*
T0*
_output_shapes
:@*
Tshape0
b
Reshape_260/shapeConst*
_output_shapes
:*
valueB"@   @   *
dtype0
l
Reshape_260Reshape
split_6:76Reshape_260/shape*
Tshape0*
T0*
_output_shapes

:@@
b
Reshape_261/shapeConst*
valueB"@   @   *
_output_shapes
:*
dtype0
l
Reshape_261Reshape
split_6:77Reshape_261/shape*
T0*
Tshape0*
_output_shapes

:@@
[
Reshape_262/shapeConst*
_output_shapes
:*
valueB:@*
dtype0
h
Reshape_262Reshape
split_6:78Reshape_262/shape*
Tshape0*
_output_shapes
:@*
T0
[
Reshape_263/shapeConst*
dtype0*
valueB:@*
_output_shapes
:
h
Reshape_263Reshape
split_6:79Reshape_263/shape*
Tshape0*
T0*
_output_shapes
:@
b
Reshape_264/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_264Reshape
split_6:80Reshape_264/shape*
Tshape0*
_output_shapes

:@*
T0
b
Reshape_265/shapeConst*
_output_shapes
:*
dtype0*
valueB"@      
l
Reshape_265Reshape
split_6:81Reshape_265/shape*
_output_shapes

:@*
Tshape0*
T0
[
Reshape_266/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_266Reshape
split_6:82Reshape_266/shape*
Tshape0*
_output_shapes
:*
T0
[
Reshape_267/shapeConst*
dtype0*
valueB:*
_output_shapes
:
h
Reshape_267Reshape
split_6:83Reshape_267/shape*
T0*
_output_shapes
:*
Tshape0
?
	Assign_25Assignpi/dense/kernelReshape_184*
_output_shapes

:<@*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(
?
	Assign_26Assignpi/dense/biasReshape_185* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
?
	Assign_27Assignpi/dense_1/kernelReshape_186*
T0*
_output_shapes

:@@*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
?
	Assign_28Assignpi/dense_1/biasReshape_187*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
?
	Assign_29Assignpi/dense_2/kernelReshape_188*
_output_shapes

:@*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0
?
	Assign_30Assignpi/dense_2/biasReshape_189*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias
?
	Assign_31Assign
pi/log_stdReshape_190*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std*
validate_shape(
?
	Assign_32Assignvf/dense/kernelReshape_191*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes

:<@*
validate_shape(
?
	Assign_33Assignvf/dense/biasReshape_192*
use_locking(*
_output_shapes
:@*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
?
	Assign_34Assignvf/dense_1/kernelReshape_193*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
_output_shapes

:@@*
T0
?
	Assign_35Assignvf/dense_1/biasReshape_194*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*"
_class
loc:@vf/dense_1/bias
?
	Assign_36Assignvf/dense_2/kernelReshape_195*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes

:@*
validate_shape(*
T0
?
	Assign_37Assignvf/dense_2/biasReshape_196*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@vf/dense_2/bias
?
	Assign_38Assignvc/dense/kernelReshape_197*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes

:<@*
validate_shape(*
T0
?
	Assign_39Assignvc/dense/biasReshape_198* 
_class
loc:@vc/dense/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
?
	Assign_40Assignvc/dense_1/kernelReshape_199*
validate_shape(*
_output_shapes

:@@*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0
?
	Assign_41Assignvc/dense_1/biasReshape_200*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes
:@
?
	Assign_42Assignvc/dense_2/kernelReshape_201*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(
?
	Assign_43Assignvc/dense_2/biasReshape_202*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
?
	Assign_44Assignlam/dense/kernelReshape_203*
validate_shape(*
T0*#
_class
loc:@lam/dense/kernel*
_output_shapes

:<@*
use_locking(
?
	Assign_45Assignlam/dense/biasReshape_204*
T0*!
_class
loc:@lam/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
?
	Assign_46Assignlam/dense_1/kernelReshape_205*%
_class
loc:@lam/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0*
use_locking(
?
	Assign_47Assignlam/dense_1/biasReshape_206*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*#
_class
loc:@lam/dense_1/bias
?
	Assign_48Assignlam/dense_2/kernelReshape_207*
validate_shape(*%
_class
loc:@lam/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@
?
	Assign_49Assignlam/dense_2/biasReshape_208*#
_class
loc:@lam/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
?
	Assign_50AssignVariableReshape_209*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@Variable*
validate_shape(
?
	Assign_51Assignbeta1_powerReshape_210*
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
?
	Assign_52Assignbeta2_powerReshape_211*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
?
	Assign_53Assignpi/dense/kernel/AdamReshape_212*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:<@*
use_locking(
?
	Assign_54Assignpi/dense/kernel/Adam_1Reshape_213*
T0*
validate_shape(*
_output_shapes

:<@*"
_class
loc:@pi/dense/kernel*
use_locking(
?
	Assign_55Assignpi/dense/bias/AdamReshape_214*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias
?
	Assign_56Assignpi/dense/bias/Adam_1Reshape_215*
_output_shapes
:@*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
?
	Assign_57Assignpi/dense_1/kernel/AdamReshape_216*
_output_shapes

:@@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
?
	Assign_58Assignpi/dense_1/kernel/Adam_1Reshape_217*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@@
?
	Assign_59Assignpi/dense_1/bias/AdamReshape_218*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
?
	Assign_60Assignpi/dense_1/bias/Adam_1Reshape_219*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0
?
	Assign_61Assignpi/dense_2/kernel/AdamReshape_220*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
?
	Assign_62Assignpi/dense_2/kernel/Adam_1Reshape_221*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0
?
	Assign_63Assignpi/dense_2/bias/AdamReshape_222*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
	Assign_64Assignpi/dense_2/bias/Adam_1Reshape_223*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
?
	Assign_65Assignpi/log_std/AdamReshape_224*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
T0*
use_locking(
?
	Assign_66Assignpi/log_std/Adam_1Reshape_225*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(*
_output_shapes
:
?
	Assign_67Assign
Variable_1Reshape_226*
_class
loc:@Variable_1*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
?
	Assign_68Assignbeta1_power_1Reshape_227*
use_locking(*
_output_shapes
: *!
_class
loc:@lam/dense/bias*
T0*
validate_shape(
?
	Assign_69Assignbeta2_power_1Reshape_228*
T0*!
_class
loc:@lam/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
?
	Assign_70Assignlam/dense/kernel/AdamReshape_229*
_output_shapes

:<@*
use_locking(*#
_class
loc:@lam/dense/kernel*
T0*
validate_shape(
?
	Assign_71Assignlam/dense/kernel/Adam_1Reshape_230*
validate_shape(*#
_class
loc:@lam/dense/kernel*
_output_shapes

:<@*
T0*
use_locking(
?
	Assign_72Assignlam/dense/bias/AdamReshape_231*
use_locking(*
validate_shape(*!
_class
loc:@lam/dense/bias*
T0*
_output_shapes
:@
?
	Assign_73Assignlam/dense/bias/Adam_1Reshape_232*!
_class
loc:@lam/dense/bias*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
?
	Assign_74Assignlam/dense_1/kernel/AdamReshape_233*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel
?
	Assign_75Assignlam/dense_1/kernel/Adam_1Reshape_234*
_output_shapes

:@@*
validate_shape(*
use_locking(*%
_class
loc:@lam/dense_1/kernel*
T0
?
	Assign_76Assignlam/dense_1/bias/AdamReshape_235*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*#
_class
loc:@lam/dense_1/bias
?
	Assign_77Assignlam/dense_1/bias/Adam_1Reshape_236*
T0*
_output_shapes
:@*
validate_shape(*#
_class
loc:@lam/dense_1/bias*
use_locking(
?
	Assign_78Assignlam/dense_2/kernel/AdamReshape_237*
T0*
use_locking(*
validate_shape(*%
_class
loc:@lam/dense_2/kernel*
_output_shapes

:@
?
	Assign_79Assignlam/dense_2/kernel/Adam_1Reshape_238*
T0*%
_class
loc:@lam/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@
?
	Assign_80Assignlam/dense_2/bias/AdamReshape_239*
validate_shape(*
_output_shapes
:*#
_class
loc:@lam/dense_2/bias*
use_locking(*
T0
?
	Assign_81Assignlam/dense_2/bias/Adam_1Reshape_240*
_output_shapes
:*
use_locking(*#
_class
loc:@lam/dense_2/bias*
validate_shape(*
T0
?
	Assign_82Assign
Variable_2Reshape_241*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@Variable_2*
validate_shape(
?
	Assign_83Assignbeta1_power_2Reshape_242* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
?
	Assign_84Assignbeta2_power_2Reshape_243*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
?
	Assign_85Assignvf/dense/kernel/AdamReshape_244*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes

:<@
?
	Assign_86Assignvf/dense/kernel/Adam_1Reshape_245*
T0*"
_class
loc:@vf/dense/kernel*
validate_shape(*
_output_shapes

:<@*
use_locking(
?
	Assign_87Assignvf/dense/bias/AdamReshape_246*
T0*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(
?
	Assign_88Assignvf/dense/bias/Adam_1Reshape_247*
T0*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(
?
	Assign_89Assignvf/dense_1/kernel/AdamReshape_248*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
_output_shapes

:@@*
T0
?
	Assign_90Assignvf/dense_1/kernel/Adam_1Reshape_249*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
_output_shapes

:@@*
T0
?
	Assign_91Assignvf/dense_1/bias/AdamReshape_250*
validate_shape(*
_output_shapes
:@*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0
?
	Assign_92Assignvf/dense_1/bias/Adam_1Reshape_251*
T0*
_output_shapes
:@*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(
?
	Assign_93Assignvf/dense_2/kernel/AdamReshape_252*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
?
	Assign_94Assignvf/dense_2/kernel/Adam_1Reshape_253*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes

:@*
T0*
use_locking(
?
	Assign_95Assignvf/dense_2/bias/AdamReshape_254*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0
?
	Assign_96Assignvf/dense_2/bias/Adam_1Reshape_255*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
?
	Assign_97Assignvc/dense/kernel/AdamReshape_256*
_output_shapes

:<@*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel
?
	Assign_98Assignvc/dense/kernel/Adam_1Reshape_257*"
_class
loc:@vc/dense/kernel*
_output_shapes

:<@*
use_locking(*
T0*
validate_shape(
?
	Assign_99Assignvc/dense/bias/AdamReshape_258*
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
_output_shapes
:@
?

Assign_100Assignvc/dense/bias/Adam_1Reshape_259*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
:@
?

Assign_101Assignvc/dense_1/kernel/AdamReshape_260*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@vc/dense_1/kernel
?

Assign_102Assignvc/dense_1/kernel/Adam_1Reshape_261*
_output_shapes

:@@*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
validate_shape(*
T0
?

Assign_103Assignvc/dense_1/bias/AdamReshape_262*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes
:@
?

Assign_104Assignvc/dense_1/bias/Adam_1Reshape_263*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*"
_class
loc:@vc/dense_1/bias
?

Assign_105Assignvc/dense_2/kernel/AdamReshape_264*
_output_shapes

:@*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
?

Assign_106Assignvc/dense_2/kernel/Adam_1Reshape_265*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
?

Assign_107Assignvc/dense_2/bias/AdamReshape_266*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(
?

Assign_108Assignvc/dense_2/bias/Adam_1Reshape_267*"
_class
loc:@vc/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
?
group_deps_6NoOp^Assign_100^Assign_101^Assign_102^Assign_103^Assign_104^Assign_105^Assign_106^Assign_107^Assign_108
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65
^Assign_66
^Assign_67
^Assign_68
^Assign_69
^Assign_70
^Assign_71
^Assign_72
^Assign_73
^Assign_74
^Assign_75
^Assign_76
^Assign_77
^Assign_78
^Assign_79
^Assign_80
^Assign_81
^Assign_82
^Assign_83
^Assign_84
^Assign_85
^Assign_86
^Assign_87
^Assign_88
^Assign_89
^Assign_90
^Assign_91
^Assign_92
^Assign_93
^Assign_94
^Assign_95
^Assign_96
^Assign_97
^Assign_98
^Assign_99
Y
save/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
_output_shapes
: *
dtype0
?
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_99596f666de1411383c446f43acf7355/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*
_output_shapes
:T*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
?
save/SaveV2/shape_and_slicesConst*
dtype0*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:T
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2beta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2lam/dense/biaslam/dense/bias/Adamlam/dense/bias/Adam_1lam/dense/kernellam/dense/kernel/Adamlam/dense/kernel/Adam_1lam/dense_1/biaslam/dense_1/bias/Adamlam/dense_1/bias/Adam_1lam/dense_1/kernellam/dense_1/kernel/Adamlam/dense_1/kernel/Adam_1lam/dense_2/biaslam/dense_2/bias/Adamlam/dense_2/bias/Adam_1lam/dense_2/kernellam/dense_2/kernel/Adamlam/dense_2/kernel/Adam_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*b
dtypesX
V2T
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*

axis *
N*
T0
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
?
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:T*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
?
save/RestoreV2/shape_and_slicesConst*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:T
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*b
dtypesX
V2T*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
_class
loc:@Variable*
validate_shape(*
T0*
_output_shapes
: 
?
save/Assign_1Assign
Variable_1save/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@Variable_1*
use_locking(
?
save/Assign_2Assign
Variable_2save/RestoreV2:2*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@Variable_2
?
save/Assign_3Assignbeta1_powersave/RestoreV2:3*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
: 
?
save/Assign_4Assignbeta1_power_1save/RestoreV2:4*
use_locking(*
validate_shape(*!
_class
loc:@lam/dense/bias*
T0*
_output_shapes
: 
?
save/Assign_5Assignbeta1_power_2save/RestoreV2:5* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
?
save/Assign_6Assignbeta2_powersave/RestoreV2:6* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
?
save/Assign_7Assignbeta2_power_1save/RestoreV2:7*
validate_shape(*!
_class
loc:@lam/dense/bias*
_output_shapes
: *
T0*
use_locking(
?
save/Assign_8Assignbeta2_power_2save/RestoreV2:8*
T0*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(
?
save/Assign_9Assignlam/dense/biassave/RestoreV2:9*
use_locking(*
T0*
validate_shape(*!
_class
loc:@lam/dense/bias*
_output_shapes
:@
?
save/Assign_10Assignlam/dense/bias/Adamsave/RestoreV2:10*
T0*!
_class
loc:@lam/dense/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
?
save/Assign_11Assignlam/dense/bias/Adam_1save/RestoreV2:11*
validate_shape(*
T0*!
_class
loc:@lam/dense/bias*
use_locking(*
_output_shapes
:@
?
save/Assign_12Assignlam/dense/kernelsave/RestoreV2:12*
T0*
validate_shape(*
use_locking(*
_output_shapes

:<@*#
_class
loc:@lam/dense/kernel
?
save/Assign_13Assignlam/dense/kernel/Adamsave/RestoreV2:13*#
_class
loc:@lam/dense/kernel*
validate_shape(*
_output_shapes

:<@*
T0*
use_locking(
?
save/Assign_14Assignlam/dense/kernel/Adam_1save/RestoreV2:14*#
_class
loc:@lam/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:<@
?
save/Assign_15Assignlam/dense_1/biassave/RestoreV2:15*
use_locking(*
validate_shape(*#
_class
loc:@lam/dense_1/bias*
T0*
_output_shapes
:@
?
save/Assign_16Assignlam/dense_1/bias/Adamsave/RestoreV2:16*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*#
_class
loc:@lam/dense_1/bias
?
save/Assign_17Assignlam/dense_1/bias/Adam_1save/RestoreV2:17*
use_locking(*#
_class
loc:@lam/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
?
save/Assign_18Assignlam/dense_1/kernelsave/RestoreV2:18*
T0*
use_locking(*
validate_shape(*%
_class
loc:@lam/dense_1/kernel*
_output_shapes

:@@
?
save/Assign_19Assignlam/dense_1/kernel/Adamsave/RestoreV2:19*%
_class
loc:@lam/dense_1/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@
?
save/Assign_20Assignlam/dense_1/kernel/Adam_1save/RestoreV2:20*
T0*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel*
use_locking(*
validate_shape(
?
save/Assign_21Assignlam/dense_2/biassave/RestoreV2:21*
validate_shape(*
T0*#
_class
loc:@lam/dense_2/bias*
use_locking(*
_output_shapes
:
?
save/Assign_22Assignlam/dense_2/bias/Adamsave/RestoreV2:22*
validate_shape(*#
_class
loc:@lam/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
?
save/Assign_23Assignlam/dense_2/bias/Adam_1save/RestoreV2:23*
_output_shapes
:*
use_locking(*#
_class
loc:@lam/dense_2/bias*
T0*
validate_shape(
?
save/Assign_24Assignlam/dense_2/kernelsave/RestoreV2:24*
use_locking(*%
_class
loc:@lam/dense_2/kernel*
T0*
validate_shape(*
_output_shapes

:@
?
save/Assign_25Assignlam/dense_2/kernel/Adamsave/RestoreV2:25*%
_class
loc:@lam/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(*
T0
?
save/Assign_26Assignlam/dense_2/kernel/Adam_1save/RestoreV2:26*
use_locking(*%
_class
loc:@lam/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0
?
save/Assign_27Assignpi/dense/biassave/RestoreV2:27* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
?
save/Assign_28Assignpi/dense/bias/Adamsave/RestoreV2:28*
_output_shapes
:@*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
?
save/Assign_29Assignpi/dense/bias/Adam_1save/RestoreV2:29* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
?
save/Assign_30Assignpi/dense/kernelsave/RestoreV2:30*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:<@
?
save/Assign_31Assignpi/dense/kernel/Adamsave/RestoreV2:31*
T0*
_output_shapes

:<@*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
?
save/Assign_32Assignpi/dense/kernel/Adam_1save/RestoreV2:32*
_output_shapes

:<@*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel
?
save/Assign_33Assignpi/dense_1/biassave/RestoreV2:33*
use_locking(*
_output_shapes
:@*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
?
save/Assign_34Assignpi/dense_1/bias/Adamsave/RestoreV2:34*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(
?
save/Assign_35Assignpi/dense_1/bias/Adam_1save/RestoreV2:35*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*"
_class
loc:@pi/dense_1/bias
?
save/Assign_36Assignpi/dense_1/kernelsave/RestoreV2:36*
_output_shapes

:@@*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel
?
save/Assign_37Assignpi/dense_1/kernel/Adamsave/RestoreV2:37*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
?
save/Assign_38Assignpi/dense_1/kernel/Adam_1save/RestoreV2:38*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
_output_shapes

:@@
?
save/Assign_39Assignpi/dense_2/biassave/RestoreV2:39*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
?
save/Assign_40Assignpi/dense_2/bias/Adamsave/RestoreV2:40*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias
?
save/Assign_41Assignpi/dense_2/bias/Adam_1save/RestoreV2:41*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
?
save/Assign_42Assignpi/dense_2/kernelsave/RestoreV2:42*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel
?
save/Assign_43Assignpi/dense_2/kernel/Adamsave/RestoreV2:43*
_output_shapes

:@*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0
?
save/Assign_44Assignpi/dense_2/kernel/Adam_1save/RestoreV2:44*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel
?
save/Assign_45Assign
pi/log_stdsave/RestoreV2:45*
use_locking(*
_output_shapes
:*
validate_shape(*
_class
loc:@pi/log_std*
T0
?
save/Assign_46Assignpi/log_std/Adamsave/RestoreV2:46*
T0*
validate_shape(*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(
?
save/Assign_47Assignpi/log_std/Adam_1save/RestoreV2:47*
_class
loc:@pi/log_std*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
?
save/Assign_48Assignvc/dense/biassave/RestoreV2:48* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
?
save/Assign_49Assignvc/dense/bias/Adamsave/RestoreV2:49*
T0*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
:@
?
save/Assign_50Assignvc/dense/bias/Adam_1save/RestoreV2:50*
_output_shapes
:@*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(*
T0
?
save/Assign_51Assignvc/dense/kernelsave/RestoreV2:51*
use_locking(*
T0*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
validate_shape(
?
save/Assign_52Assignvc/dense/kernel/Adamsave/RestoreV2:52*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(*
use_locking(
?
save/Assign_53Assignvc/dense/kernel/Adam_1save/RestoreV2:53*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes

:<@
?
save/Assign_54Assignvc/dense_1/biassave/RestoreV2:54*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias
?
save/Assign_55Assignvc/dense_1/bias/Adamsave/RestoreV2:55*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_1/bias
?
save/Assign_56Assignvc/dense_1/bias/Adam_1save/RestoreV2:56*"
_class
loc:@vc/dense_1/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
?
save/Assign_57Assignvc/dense_1/kernelsave/RestoreV2:57*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
_output_shapes

:@@*
T0*
use_locking(
?
save/Assign_58Assignvc/dense_1/kernel/Adamsave/RestoreV2:58*
T0*
_output_shapes

:@@*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_1/kernel
?
save/Assign_59Assignvc/dense_1/kernel/Adam_1save/RestoreV2:59*
_output_shapes

:@@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_1/kernel
?
save/Assign_60Assignvc/dense_2/biassave/RestoreV2:60*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(
?
save/Assign_61Assignvc/dense_2/bias/Adamsave/RestoreV2:61*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
?
save/Assign_62Assignvc/dense_2/bias/Adam_1save/RestoreV2:62*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
?
save/Assign_63Assignvc/dense_2/kernelsave/RestoreV2:63*
T0*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes

:@*
validate_shape(
?
save/Assign_64Assignvc/dense_2/kernel/Adamsave/RestoreV2:64*
T0*
_output_shapes

:@*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
validate_shape(
?
save/Assign_65Assignvc/dense_2/kernel/Adam_1save/RestoreV2:65*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
?
save/Assign_66Assignvf/dense/biassave/RestoreV2:66*
_output_shapes
:@*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0
?
save/Assign_67Assignvf/dense/bias/Adamsave/RestoreV2:67*
use_locking(*
_output_shapes
:@*
T0* 
_class
loc:@vf/dense/bias*
validate_shape(
?
save/Assign_68Assignvf/dense/bias/Adam_1save/RestoreV2:68* 
_class
loc:@vf/dense/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
?
save/Assign_69Assignvf/dense/kernelsave/RestoreV2:69*
_output_shapes

:<@*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0
?
save/Assign_70Assignvf/dense/kernel/Adamsave/RestoreV2:70*
_output_shapes

:<@*"
_class
loc:@vf/dense/kernel*
T0*
use_locking(*
validate_shape(
?
save/Assign_71Assignvf/dense/kernel/Adam_1save/RestoreV2:71*
validate_shape(*
_output_shapes

:<@*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
?
save/Assign_72Assignvf/dense_1/biassave/RestoreV2:72*
use_locking(*
_output_shapes
:@*
T0*"
_class
loc:@vf/dense_1/bias*
validate_shape(
?
save/Assign_73Assignvf/dense_1/bias/Adamsave/RestoreV2:73*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
_output_shapes
:@
?
save/Assign_74Assignvf/dense_1/bias/Adam_1save/RestoreV2:74*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes
:@*
T0
?
save/Assign_75Assignvf/dense_1/kernelsave/RestoreV2:75*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
use_locking(
?
save/Assign_76Assignvf/dense_1/kernel/Adamsave/RestoreV2:76*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
T0*
_output_shapes

:@@
?
save/Assign_77Assignvf/dense_1/kernel/Adam_1save/RestoreV2:77*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
_output_shapes

:@@*
use_locking(
?
save/Assign_78Assignvf/dense_2/biassave/RestoreV2:78*
use_locking(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0*
validate_shape(
?
save/Assign_79Assignvf/dense_2/bias/Adamsave/RestoreV2:79*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
?
save/Assign_80Assignvf/dense_2/bias/Adam_1save/RestoreV2:80*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias
?
save/Assign_81Assignvf/dense_2/kernelsave/RestoreV2:81*
_output_shapes

:@*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_2/kernel
?
save/Assign_82Assignvf/dense_2/kernel/Adamsave/RestoreV2:82*
T0*$
_class
loc:@vf/dense_2/kernel*
_output_shapes

:@*
use_locking(*
validate_shape(
?
save/Assign_83Assignvf/dense_2/kernel/Adam_1save/RestoreV2:83*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 
?
save_1/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_c539bb1599d948888275c2e3b935d7cf/part*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:T
?
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:T*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2beta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2lam/dense/biaslam/dense/bias/Adamlam/dense/bias/Adam_1lam/dense/kernellam/dense/kernel/Adamlam/dense/kernel/Adam_1lam/dense_1/biaslam/dense_1/bias/Adamlam/dense_1/bias/Adam_1lam/dense_1/kernellam/dense_1/kernel/Adamlam/dense_1/kernel/Adam_1lam/dense_2/biaslam/dense_2/bias/Adamlam/dense_2/bias/Adam_1lam/dense_2/kernellam/dense_2/kernel/Adamlam/dense_2/kernel/Adam_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*b
dtypesX
V2T
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*

axis *
_output_shapes
:*
T0
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst*
_output_shapes
:T*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0
?
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:T*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T
?
save_1/AssignAssignVariablesave_1/RestoreV2*
use_locking(*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: *
T0
?
save_1/Assign_1Assign
Variable_1save_1/RestoreV2:1*
_class
loc:@Variable_1*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
?
save_1/Assign_2Assign
Variable_2save_1/RestoreV2:2*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@Variable_2
?
save_1/Assign_3Assignbeta1_powersave_1/RestoreV2:3*
T0*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
?
save_1/Assign_4Assignbeta1_power_1save_1/RestoreV2:4*
validate_shape(*!
_class
loc:@lam/dense/bias*
_output_shapes
: *
T0*
use_locking(
?
save_1/Assign_5Assignbeta1_power_2save_1/RestoreV2:5*
validate_shape(*
_output_shapes
: *
T0*
use_locking(* 
_class
loc:@vc/dense/bias
?
save_1/Assign_6Assignbeta2_powersave_1/RestoreV2:6*
use_locking(*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
?
save_1/Assign_7Assignbeta2_power_1save_1/RestoreV2:7*
use_locking(*
T0*
_output_shapes
: *!
_class
loc:@lam/dense/bias*
validate_shape(
?
save_1/Assign_8Assignbeta2_power_2save_1/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
validate_shape(
?
save_1/Assign_9Assignlam/dense/biassave_1/RestoreV2:9*!
_class
loc:@lam/dense/bias*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(
?
save_1/Assign_10Assignlam/dense/bias/Adamsave_1/RestoreV2:10*
validate_shape(*
use_locking(*
T0*!
_class
loc:@lam/dense/bias*
_output_shapes
:@
?
save_1/Assign_11Assignlam/dense/bias/Adam_1save_1/RestoreV2:11*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*!
_class
loc:@lam/dense/bias
?
save_1/Assign_12Assignlam/dense/kernelsave_1/RestoreV2:12*
validate_shape(*
use_locking(*
_output_shapes

:<@*
T0*#
_class
loc:@lam/dense/kernel
?
save_1/Assign_13Assignlam/dense/kernel/Adamsave_1/RestoreV2:13*#
_class
loc:@lam/dense/kernel*
_output_shapes

:<@*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_14Assignlam/dense/kernel/Adam_1save_1/RestoreV2:14*
_output_shapes

:<@*
use_locking(*#
_class
loc:@lam/dense/kernel*
validate_shape(*
T0
?
save_1/Assign_15Assignlam/dense_1/biassave_1/RestoreV2:15*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*#
_class
loc:@lam/dense_1/bias
?
save_1/Assign_16Assignlam/dense_1/bias/Adamsave_1/RestoreV2:16*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*#
_class
loc:@lam/dense_1/bias
?
save_1/Assign_17Assignlam/dense_1/bias/Adam_1save_1/RestoreV2:17*
validate_shape(*
use_locking(*
_output_shapes
:@*#
_class
loc:@lam/dense_1/bias*
T0
?
save_1/Assign_18Assignlam/dense_1/kernelsave_1/RestoreV2:18*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_19Assignlam/dense_1/kernel/Adamsave_1/RestoreV2:19*
T0*
_output_shapes

:@@*
validate_shape(*%
_class
loc:@lam/dense_1/kernel*
use_locking(
?
save_1/Assign_20Assignlam/dense_1/kernel/Adam_1save_1/RestoreV2:20*
_output_shapes

:@@*
validate_shape(*
use_locking(*
T0*%
_class
loc:@lam/dense_1/kernel
?
save_1/Assign_21Assignlam/dense_2/biassave_1/RestoreV2:21*
T0*#
_class
loc:@lam/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
?
save_1/Assign_22Assignlam/dense_2/bias/Adamsave_1/RestoreV2:22*
T0*
_output_shapes
:*#
_class
loc:@lam/dense_2/bias*
use_locking(*
validate_shape(
?
save_1/Assign_23Assignlam/dense_2/bias/Adam_1save_1/RestoreV2:23*
T0*#
_class
loc:@lam/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
?
save_1/Assign_24Assignlam/dense_2/kernelsave_1/RestoreV2:24*
_output_shapes

:@*
T0*
validate_shape(*%
_class
loc:@lam/dense_2/kernel*
use_locking(
?
save_1/Assign_25Assignlam/dense_2/kernel/Adamsave_1/RestoreV2:25*
_output_shapes

:@*
T0*
use_locking(*%
_class
loc:@lam/dense_2/kernel*
validate_shape(
?
save_1/Assign_26Assignlam/dense_2/kernel/Adam_1save_1/RestoreV2:26*
T0*%
_class
loc:@lam/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@
?
save_1/Assign_27Assignpi/dense/biassave_1/RestoreV2:27* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
?
save_1/Assign_28Assignpi/dense/bias/Adamsave_1/RestoreV2:28*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(
?
save_1/Assign_29Assignpi/dense/bias/Adam_1save_1/RestoreV2:29*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
use_locking(*
T0
?
save_1/Assign_30Assignpi/dense/kernelsave_1/RestoreV2:30*
_output_shapes

:<@*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
?
save_1/Assign_31Assignpi/dense/kernel/Adamsave_1/RestoreV2:31*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:<@*
use_locking(
?
save_1/Assign_32Assignpi/dense/kernel/Adam_1save_1/RestoreV2:32*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:<@
?
save_1/Assign_33Assignpi/dense_1/biassave_1/RestoreV2:33*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
?
save_1/Assign_34Assignpi/dense_1/bias/Adamsave_1/RestoreV2:34*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_35Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:35*
_output_shapes
:@*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(
?
save_1/Assign_36Assignpi/dense_1/kernelsave_1/RestoreV2:36*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@*
use_locking(*
validate_shape(
?
save_1/Assign_37Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:37*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
use_locking(*
T0
?
save_1/Assign_38Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:38*
_output_shapes

:@@*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0
?
save_1/Assign_39Assignpi/dense_2/biassave_1/RestoreV2:39*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
?
save_1/Assign_40Assignpi/dense_2/bias/Adamsave_1/RestoreV2:40*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
?
save_1/Assign_41Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:41*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
?
save_1/Assign_42Assignpi/dense_2/kernelsave_1/RestoreV2:42*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@
?
save_1/Assign_43Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:43*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
?
save_1/Assign_44Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:44*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(
?
save_1/Assign_45Assign
pi/log_stdsave_1/RestoreV2:45*
_output_shapes
:*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(
?
save_1/Assign_46Assignpi/log_std/Adamsave_1/RestoreV2:46*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
?
save_1/Assign_47Assignpi/log_std/Adam_1save_1/RestoreV2:47*
T0*
_class
loc:@pi/log_std*
validate_shape(*
_output_shapes
:*
use_locking(
?
save_1/Assign_48Assignvc/dense/biassave_1/RestoreV2:48*
use_locking(*
validate_shape(*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
T0
?
save_1/Assign_49Assignvc/dense/bias/Adamsave_1/RestoreV2:49* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
?
save_1/Assign_50Assignvc/dense/bias/Adam_1save_1/RestoreV2:50*
_output_shapes
:@*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
?
save_1/Assign_51Assignvc/dense/kernelsave_1/RestoreV2:51*
T0*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes

:<@
?
save_1/Assign_52Assignvc/dense/kernel/Adamsave_1/RestoreV2:52*
validate_shape(*
use_locking(*
T0*
_output_shapes

:<@*"
_class
loc:@vc/dense/kernel
?
save_1/Assign_53Assignvc/dense/kernel/Adam_1save_1/RestoreV2:53*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
use_locking(*
_output_shapes

:<@
?
save_1/Assign_54Assignvc/dense_1/biassave_1/RestoreV2:54*
T0*
_output_shapes
:@*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(
?
save_1/Assign_55Assignvc/dense_1/bias/Adamsave_1/RestoreV2:55*
_output_shapes
:@*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(*
use_locking(
?
save_1/Assign_56Assignvc/dense_1/bias/Adam_1save_1/RestoreV2:56*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*"
_class
loc:@vc/dense_1/bias
?
save_1/Assign_57Assignvc/dense_1/kernelsave_1/RestoreV2:57*$
_class
loc:@vc/dense_1/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@@
?
save_1/Assign_58Assignvc/dense_1/kernel/Adamsave_1/RestoreV2:58*
_output_shapes

:@@*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
?
save_1/Assign_59Assignvc/dense_1/kernel/Adam_1save_1/RestoreV2:59*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
_output_shapes

:@@
?
save_1/Assign_60Assignvc/dense_2/biassave_1/RestoreV2:60*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias
?
save_1/Assign_61Assignvc/dense_2/bias/Adamsave_1/RestoreV2:61*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:
?
save_1/Assign_62Assignvc/dense_2/bias/Adam_1save_1/RestoreV2:62*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
?
save_1/Assign_63Assignvc/dense_2/kernelsave_1/RestoreV2:63*
use_locking(*
_output_shapes

:@*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
?
save_1/Assign_64Assignvc/dense_2/kernel/Adamsave_1/RestoreV2:64*
_output_shapes

:@*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
T0
?
save_1/Assign_65Assignvc/dense_2/kernel/Adam_1save_1/RestoreV2:65*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes

:@*
validate_shape(
?
save_1/Assign_66Assignvf/dense/biassave_1/RestoreV2:66* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes
:@*
T0*
use_locking(
?
save_1/Assign_67Assignvf/dense/bias/Adamsave_1/RestoreV2:67*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
use_locking(
?
save_1/Assign_68Assignvf/dense/bias/Adam_1save_1/RestoreV2:68*
T0*
_output_shapes
:@*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(
?
save_1/Assign_69Assignvf/dense/kernelsave_1/RestoreV2:69*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes

:<@*
use_locking(*
validate_shape(
?
save_1/Assign_70Assignvf/dense/kernel/Adamsave_1/RestoreV2:70*
validate_shape(*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes

:<@*
use_locking(
?
save_1/Assign_71Assignvf/dense/kernel/Adam_1save_1/RestoreV2:71*
_output_shapes

:<@*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
validate_shape(
?
save_1/Assign_72Assignvf/dense_1/biassave_1/RestoreV2:72*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(*
_output_shapes
:@*
validate_shape(
?
save_1/Assign_73Assignvf/dense_1/bias/Adamsave_1/RestoreV2:73*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(*"
_class
loc:@vf/dense_1/bias
?
save_1/Assign_74Assignvf/dense_1/bias/Adam_1save_1/RestoreV2:74*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes
:@*
T0
?
save_1/Assign_75Assignvf/dense_1/kernelsave_1/RestoreV2:75*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
T0
?
save_1/Assign_76Assignvf/dense_1/kernel/Adamsave_1/RestoreV2:76*
_output_shapes

:@@*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel
?
save_1/Assign_77Assignvf/dense_1/kernel/Adam_1save_1/RestoreV2:77*
T0*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(
?
save_1/Assign_78Assignvf/dense_2/biassave_1/RestoreV2:78*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_79Assignvf/dense_2/bias/Adamsave_1/RestoreV2:79*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_80Assignvf/dense_2/bias/Adam_1save_1/RestoreV2:80*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_1/Assign_81Assignvf/dense_2/kernelsave_1/RestoreV2:81*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@vf/dense_2/kernel
?
save_1/Assign_82Assignvf/dense_2/kernel/Adamsave_1/RestoreV2:82*
T0*
_output_shapes

:@*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(
?
save_1/Assign_83Assignvf/dense_2/kernel/Adam_1save_1/RestoreV2:83*
_output_shapes

:@*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
_output_shapes
: *
shape: *
dtype0
?
save_2/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_7375eb74707e46818b740682e1130bbe/part
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
?
save_2/SaveV2/tensor_namesConst*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:T*
dtype0
?
save_2/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:T*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2beta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2lam/dense/biaslam/dense/bias/Adamlam/dense/bias/Adam_1lam/dense/kernellam/dense/kernel/Adamlam/dense/kernel/Adam_1lam/dense_1/biaslam/dense_1/bias/Adamlam/dense_1/bias/Adam_1lam/dense_1/kernellam/dense_1/kernel/Adamlam/dense_1/kernel/Adam_1lam/dense_2/biaslam/dense_2/bias/Adamlam/dense_2/bias/Adam_1lam/dense_2/kernellam/dense_2/kernel/Adamlam/dense_2/kernel/Adam_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*b
dtypesX
V2T
?
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
_output_shapes
: *)
_class
loc:@save_2/ShardedFilename*
T0
?
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*

axis *
T0*
_output_shapes
:
?
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
?
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
?
save_2/RestoreV2/tensor_namesConst*
_output_shapes
:T*
dtype0*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
?
!save_2/RestoreV2/shape_and_slicesConst*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:T
?
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T
?
save_2/AssignAssignVariablesave_2/RestoreV2*
_class
loc:@Variable*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
?
save_2/Assign_1Assign
Variable_1save_2/RestoreV2:1*
use_locking(*
_class
loc:@Variable_1*
_output_shapes
: *
validate_shape(*
T0
?
save_2/Assign_2Assign
Variable_2save_2/RestoreV2:2*
T0*
use_locking(*
_class
loc:@Variable_2*
_output_shapes
: *
validate_shape(
?
save_2/Assign_3Assignbeta1_powersave_2/RestoreV2:3* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
?
save_2/Assign_4Assignbeta1_power_1save_2/RestoreV2:4*
T0*
_output_shapes
: *
use_locking(*!
_class
loc:@lam/dense/bias*
validate_shape(
?
save_2/Assign_5Assignbeta1_power_2save_2/RestoreV2:5*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(
?
save_2/Assign_6Assignbeta2_powersave_2/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
?
save_2/Assign_7Assignbeta2_power_1save_2/RestoreV2:7*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*!
_class
loc:@lam/dense/bias
?
save_2/Assign_8Assignbeta2_power_2save_2/RestoreV2:8*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@vc/dense/bias*
validate_shape(
?
save_2/Assign_9Assignlam/dense/biassave_2/RestoreV2:9*
T0*
validate_shape(*
use_locking(*!
_class
loc:@lam/dense/bias*
_output_shapes
:@
?
save_2/Assign_10Assignlam/dense/bias/Adamsave_2/RestoreV2:10*
use_locking(*
T0*
_output_shapes
:@*!
_class
loc:@lam/dense/bias*
validate_shape(
?
save_2/Assign_11Assignlam/dense/bias/Adam_1save_2/RestoreV2:11*
validate_shape(*!
_class
loc:@lam/dense/bias*
use_locking(*
_output_shapes
:@*
T0
?
save_2/Assign_12Assignlam/dense/kernelsave_2/RestoreV2:12*
_output_shapes

:<@*
use_locking(*
T0*#
_class
loc:@lam/dense/kernel*
validate_shape(
?
save_2/Assign_13Assignlam/dense/kernel/Adamsave_2/RestoreV2:13*
T0*
use_locking(*
_output_shapes

:<@*#
_class
loc:@lam/dense/kernel*
validate_shape(
?
save_2/Assign_14Assignlam/dense/kernel/Adam_1save_2/RestoreV2:14*#
_class
loc:@lam/dense/kernel*
_output_shapes

:<@*
validate_shape(*
use_locking(*
T0
?
save_2/Assign_15Assignlam/dense_1/biassave_2/RestoreV2:15*
_output_shapes
:@*
T0*#
_class
loc:@lam/dense_1/bias*
use_locking(*
validate_shape(
?
save_2/Assign_16Assignlam/dense_1/bias/Adamsave_2/RestoreV2:16*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(*#
_class
loc:@lam/dense_1/bias
?
save_2/Assign_17Assignlam/dense_1/bias/Adam_1save_2/RestoreV2:17*
_output_shapes
:@*
T0*
use_locking(*#
_class
loc:@lam/dense_1/bias*
validate_shape(
?
save_2/Assign_18Assignlam/dense_1/kernelsave_2/RestoreV2:18*%
_class
loc:@lam/dense_1/kernel*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0
?
save_2/Assign_19Assignlam/dense_1/kernel/Adamsave_2/RestoreV2:19*
use_locking(*
_output_shapes

:@@*
validate_shape(*%
_class
loc:@lam/dense_1/kernel*
T0
?
save_2/Assign_20Assignlam/dense_1/kernel/Adam_1save_2/RestoreV2:20*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel
?
save_2/Assign_21Assignlam/dense_2/biassave_2/RestoreV2:21*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*#
_class
loc:@lam/dense_2/bias
?
save_2/Assign_22Assignlam/dense_2/bias/Adamsave_2/RestoreV2:22*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*#
_class
loc:@lam/dense_2/bias
?
save_2/Assign_23Assignlam/dense_2/bias/Adam_1save_2/RestoreV2:23*
T0*
_output_shapes
:*
validate_shape(*#
_class
loc:@lam/dense_2/bias*
use_locking(
?
save_2/Assign_24Assignlam/dense_2/kernelsave_2/RestoreV2:24*
T0*
_output_shapes

:@*
validate_shape(*%
_class
loc:@lam/dense_2/kernel*
use_locking(
?
save_2/Assign_25Assignlam/dense_2/kernel/Adamsave_2/RestoreV2:25*
_output_shapes

:@*
use_locking(*%
_class
loc:@lam/dense_2/kernel*
validate_shape(*
T0
?
save_2/Assign_26Assignlam/dense_2/kernel/Adam_1save_2/RestoreV2:26*
T0*%
_class
loc:@lam/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@
?
save_2/Assign_27Assignpi/dense/biassave_2/RestoreV2:27* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(
?
save_2/Assign_28Assignpi/dense/bias/Adamsave_2/RestoreV2:28*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(
?
save_2/Assign_29Assignpi/dense/bias/Adam_1save_2/RestoreV2:29*
_output_shapes
:@*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
?
save_2/Assign_30Assignpi/dense/kernelsave_2/RestoreV2:30*
_output_shapes

:<@*"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(*
T0
?
save_2/Assign_31Assignpi/dense/kernel/Adamsave_2/RestoreV2:31*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

:<@*
use_locking(*
validate_shape(
?
save_2/Assign_32Assignpi/dense/kernel/Adam_1save_2/RestoreV2:32*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:<@
?
save_2/Assign_33Assignpi/dense_1/biassave_2/RestoreV2:33*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias
?
save_2/Assign_34Assignpi/dense_1/bias/Adamsave_2/RestoreV2:34*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
?
save_2/Assign_35Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:35*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
?
save_2/Assign_36Assignpi/dense_1/kernelsave_2/RestoreV2:36*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

:@@*
validate_shape(
?
save_2/Assign_37Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:37*
T0*
_output_shapes

:@@*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
?
save_2/Assign_38Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:38*
_output_shapes

:@@*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
?
save_2/Assign_39Assignpi/dense_2/biassave_2/RestoreV2:39*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
?
save_2/Assign_40Assignpi/dense_2/bias/Adamsave_2/RestoreV2:40*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
?
save_2/Assign_41Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:41*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0
?
save_2/Assign_42Assignpi/dense_2/kernelsave_2/RestoreV2:42*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
validate_shape(
?
save_2/Assign_43Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:43*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:@*
T0
?
save_2/Assign_44Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:44*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@
?
save_2/Assign_45Assign
pi/log_stdsave_2/RestoreV2:45*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*
_class
loc:@pi/log_std
?
save_2/Assign_46Assignpi/log_std/Adamsave_2/RestoreV2:46*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@pi/log_std
?
save_2/Assign_47Assignpi/log_std/Adam_1save_2/RestoreV2:47*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(*
T0
?
save_2/Assign_48Assignvc/dense/biassave_2/RestoreV2:48*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias
?
save_2/Assign_49Assignvc/dense/bias/Adamsave_2/RestoreV2:49*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(* 
_class
loc:@vc/dense/bias
?
save_2/Assign_50Assignvc/dense/bias/Adam_1save_2/RestoreV2:50*
use_locking(*
validate_shape(*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
T0
?
save_2/Assign_51Assignvc/dense/kernelsave_2/RestoreV2:51*
validate_shape(*
T0*
_output_shapes

:<@*
use_locking(*"
_class
loc:@vc/dense/kernel
?
save_2/Assign_52Assignvc/dense/kernel/Adamsave_2/RestoreV2:52*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes

:<@*
T0*
validate_shape(
?
save_2/Assign_53Assignvc/dense/kernel/Adam_1save_2/RestoreV2:53*
validate_shape(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes

:<@*
use_locking(
?
save_2/Assign_54Assignvc/dense_1/biassave_2/RestoreV2:54*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
_output_shapes
:@
?
save_2/Assign_55Assignvc/dense_1/bias/Adamsave_2/RestoreV2:55*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes
:@*
use_locking(
?
save_2/Assign_56Assignvc/dense_1/bias/Adam_1save_2/RestoreV2:56*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes
:@*
T0*
validate_shape(
?
save_2/Assign_57Assignvc/dense_1/kernelsave_2/RestoreV2:57*
validate_shape(*
use_locking(*
T0*
_output_shapes

:@@*$
_class
loc:@vc/dense_1/kernel
?
save_2/Assign_58Assignvc/dense_1/kernel/Adamsave_2/RestoreV2:58*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:@@
?
save_2/Assign_59Assignvc/dense_1/kernel/Adam_1save_2/RestoreV2:59*
T0*
use_locking(*
_output_shapes

:@@*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
?
save_2/Assign_60Assignvc/dense_2/biassave_2/RestoreV2:60*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
?
save_2/Assign_61Assignvc/dense_2/bias/Adamsave_2/RestoreV2:61*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias*
T0
?
save_2/Assign_62Assignvc/dense_2/bias/Adam_1save_2/RestoreV2:62*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_2/bias
?
save_2/Assign_63Assignvc/dense_2/kernelsave_2/RestoreV2:63*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes

:@
?
save_2/Assign_64Assignvc/dense_2/kernel/Adamsave_2/RestoreV2:64*
use_locking(*
T0*
_output_shapes

:@*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
?
save_2/Assign_65Assignvc/dense_2/kernel/Adam_1save_2/RestoreV2:65*
_output_shapes

:@*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel*
use_locking(
?
save_2/Assign_66Assignvf/dense/biassave_2/RestoreV2:66* 
_class
loc:@vf/dense/bias*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0
?
save_2/Assign_67Assignvf/dense/bias/Adamsave_2/RestoreV2:67*
validate_shape(*
use_locking(*
T0* 
_class
loc:@vf/dense/bias*
_output_shapes
:@
?
save_2/Assign_68Assignvf/dense/bias/Adam_1save_2/RestoreV2:68*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes
:@*
T0
?
save_2/Assign_69Assignvf/dense/kernelsave_2/RestoreV2:69*
validate_shape(*"
_class
loc:@vf/dense/kernel*
use_locking(*
T0*
_output_shapes

:<@
?
save_2/Assign_70Assignvf/dense/kernel/Adamsave_2/RestoreV2:70*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense/kernel*
_output_shapes

:<@
?
save_2/Assign_71Assignvf/dense/kernel/Adam_1save_2/RestoreV2:71*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

:<@*
T0
?
save_2/Assign_72Assignvf/dense_1/biassave_2/RestoreV2:72*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*"
_class
loc:@vf/dense_1/bias
?
save_2/Assign_73Assignvf/dense_1/bias/Adamsave_2/RestoreV2:73*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:@
?
save_2/Assign_74Assignvf/dense_1/bias/Adam_1save_2/RestoreV2:74*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
use_locking(*
T0*
_output_shapes
:@
?
save_2/Assign_75Assignvf/dense_1/kernelsave_2/RestoreV2:75*
use_locking(*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(
?
save_2/Assign_76Assignvf/dense_1/kernel/Adamsave_2/RestoreV2:76*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@
?
save_2/Assign_77Assignvf/dense_1/kernel/Adam_1save_2/RestoreV2:77*
use_locking(*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
T0
?
save_2/Assign_78Assignvf/dense_2/biassave_2/RestoreV2:78*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@vf/dense_2/bias*
T0
?
save_2/Assign_79Assignvf/dense_2/bias/Adamsave_2/RestoreV2:79*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
?
save_2/Assign_80Assignvf/dense_2/bias/Adam_1save_2/RestoreV2:80*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
?
save_2/Assign_81Assignvf/dense_2/kernelsave_2/RestoreV2:81*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(
?
save_2/Assign_82Assignvf/dense_2/kernel/Adamsave_2/RestoreV2:82*
validate_shape(*
_output_shapes

:@*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0
?
save_2/Assign_83Assignvf/dense_2/kernel/Adam_1save_2/RestoreV2:83*
_output_shapes

:@*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
use_locking(
?
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_50^save_2/Assign_51^save_2/Assign_52^save_2/Assign_53^save_2/Assign_54^save_2/Assign_55^save_2/Assign_56^save_2/Assign_57^save_2/Assign_58^save_2/Assign_59^save_2/Assign_6^save_2/Assign_60^save_2/Assign_61^save_2/Assign_62^save_2/Assign_63^save_2/Assign_64^save_2/Assign_65^save_2/Assign_66^save_2/Assign_67^save_2/Assign_68^save_2/Assign_69^save_2/Assign_7^save_2/Assign_70^save_2/Assign_71^save_2/Assign_72^save_2/Assign_73^save_2/Assign_74^save_2/Assign_75^save_2/Assign_76^save_2/Assign_77^save_2/Assign_78^save_2/Assign_79^save_2/Assign_8^save_2/Assign_80^save_2/Assign_81^save_2/Assign_82^save_2/Assign_83^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
_output_shapes
: *
dtype0*
shape: 
?
save_3/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ce929d87a4ba4d33be9de058ccf45fde/part
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_3/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
?
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
?
save_3/SaveV2/tensor_namesConst*
dtype0*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:T
?
save_3/SaveV2/shape_and_slicesConst*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:T*
dtype0
?
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2beta1_powerbeta1_power_1beta1_power_2beta2_powerbeta2_power_1beta2_power_2lam/dense/biaslam/dense/bias/Adamlam/dense/bias/Adam_1lam/dense/kernellam/dense/kernel/Adamlam/dense/kernel/Adam_1lam/dense_1/biaslam/dense_1/bias/Adamlam/dense_1/bias/Adam_1lam/dense_1/kernellam/dense_1/kernel/Adamlam/dense_1/kernel/Adam_1lam/dense_2/biaslam/dense_2/bias/Adamlam/dense_2/bias/Adam_1lam/dense_2/kernellam/dense_2/kernel/Adamlam/dense_2/kernel/Adam_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1
pi/log_stdpi/log_std/Adampi/log_std/Adam_1vc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*b
dtypesX
V2T
?
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_3/ShardedFilename
?
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
_output_shapes
:*
T0*
N*

axis 
?
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
?
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
_output_shapes
: *
T0
?
save_3/RestoreV2/tensor_namesConst*?
value?B?TBVariableB
Variable_1B
Variable_2Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta2_powerBbeta2_power_1Bbeta2_power_2Blam/dense/biasBlam/dense/bias/AdamBlam/dense/bias/Adam_1Blam/dense/kernelBlam/dense/kernel/AdamBlam/dense/kernel/Adam_1Blam/dense_1/biasBlam/dense_1/bias/AdamBlam/dense_1/bias/Adam_1Blam/dense_1/kernelBlam/dense_1/kernel/AdamBlam/dense_1/kernel/Adam_1Blam/dense_2/biasBlam/dense_2/bias/AdamBlam/dense_2/bias/Adam_1Blam/dense_2/kernelBlam/dense_2/kernel/AdamBlam/dense_2/kernel/Adam_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1B
pi/log_stdBpi/log_std/AdamBpi/log_std/Adam_1Bvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:T*
dtype0
?
!save_3/RestoreV2/shape_and_slicesConst*
dtype0*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:T
?
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T
?
save_3/AssignAssignVariablesave_3/RestoreV2*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*
_class
loc:@Variable
?
save_3/Assign_1Assign
Variable_1save_3/RestoreV2:1*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@Variable_1*
T0
?
save_3/Assign_2Assign
Variable_2save_3/RestoreV2:2*
_output_shapes
: *
use_locking(*
_class
loc:@Variable_2*
validate_shape(*
T0
?
save_3/Assign_3Assignbeta1_powersave_3/RestoreV2:3*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
T0
?
save_3/Assign_4Assignbeta1_power_1save_3/RestoreV2:4*
validate_shape(*
_output_shapes
: *
T0*!
_class
loc:@lam/dense/bias*
use_locking(
?
save_3/Assign_5Assignbeta1_power_2save_3/RestoreV2:5* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
?
save_3/Assign_6Assignbeta2_powersave_3/RestoreV2:6*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
?
save_3/Assign_7Assignbeta2_power_1save_3/RestoreV2:7*!
_class
loc:@lam/dense/bias*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
?
save_3/Assign_8Assignbeta2_power_2save_3/RestoreV2:8* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
?
save_3/Assign_9Assignlam/dense/biassave_3/RestoreV2:9*
T0*
use_locking(*!
_class
loc:@lam/dense/bias*
_output_shapes
:@*
validate_shape(
?
save_3/Assign_10Assignlam/dense/bias/Adamsave_3/RestoreV2:10*
validate_shape(*
_output_shapes
:@*!
_class
loc:@lam/dense/bias*
T0*
use_locking(
?
save_3/Assign_11Assignlam/dense/bias/Adam_1save_3/RestoreV2:11*
validate_shape(*
T0*!
_class
loc:@lam/dense/bias*
use_locking(*
_output_shapes
:@
?
save_3/Assign_12Assignlam/dense/kernelsave_3/RestoreV2:12*
T0*#
_class
loc:@lam/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

:<@
?
save_3/Assign_13Assignlam/dense/kernel/Adamsave_3/RestoreV2:13*#
_class
loc:@lam/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes

:<@*
T0
?
save_3/Assign_14Assignlam/dense/kernel/Adam_1save_3/RestoreV2:14*
T0*#
_class
loc:@lam/dense/kernel*
use_locking(*
_output_shapes

:<@*
validate_shape(
?
save_3/Assign_15Assignlam/dense_1/biassave_3/RestoreV2:15*
use_locking(*
_output_shapes
:@*
validate_shape(*#
_class
loc:@lam/dense_1/bias*
T0
?
save_3/Assign_16Assignlam/dense_1/bias/Adamsave_3/RestoreV2:16*
T0*
validate_shape(*#
_class
loc:@lam/dense_1/bias*
use_locking(*
_output_shapes
:@
?
save_3/Assign_17Assignlam/dense_1/bias/Adam_1save_3/RestoreV2:17*
T0*#
_class
loc:@lam/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
?
save_3/Assign_18Assignlam/dense_1/kernelsave_3/RestoreV2:18*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@@*%
_class
loc:@lam/dense_1/kernel
?
save_3/Assign_19Assignlam/dense_1/kernel/Adamsave_3/RestoreV2:19*
validate_shape(*
use_locking(*%
_class
loc:@lam/dense_1/kernel*
_output_shapes

:@@*
T0
?
save_3/Assign_20Assignlam/dense_1/kernel/Adam_1save_3/RestoreV2:20*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*%
_class
loc:@lam/dense_1/kernel
?
save_3/Assign_21Assignlam/dense_2/biassave_3/RestoreV2:21*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*#
_class
loc:@lam/dense_2/bias
?
save_3/Assign_22Assignlam/dense_2/bias/Adamsave_3/RestoreV2:22*
use_locking(*
T0*
validate_shape(*#
_class
loc:@lam/dense_2/bias*
_output_shapes
:
?
save_3/Assign_23Assignlam/dense_2/bias/Adam_1save_3/RestoreV2:23*
validate_shape(*#
_class
loc:@lam/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
?
save_3/Assign_24Assignlam/dense_2/kernelsave_3/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes

:@*%
_class
loc:@lam/dense_2/kernel*
T0
?
save_3/Assign_25Assignlam/dense_2/kernel/Adamsave_3/RestoreV2:25*
T0*
validate_shape(*%
_class
loc:@lam/dense_2/kernel*
_output_shapes

:@*
use_locking(
?
save_3/Assign_26Assignlam/dense_2/kernel/Adam_1save_3/RestoreV2:26*
T0*
_output_shapes

:@*
use_locking(*
validate_shape(*%
_class
loc:@lam/dense_2/kernel
?
save_3/Assign_27Assignpi/dense/biassave_3/RestoreV2:27*
T0*
validate_shape(*
_output_shapes
:@* 
_class
loc:@pi/dense/bias*
use_locking(
?
save_3/Assign_28Assignpi/dense/bias/Adamsave_3/RestoreV2:28*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
?
save_3/Assign_29Assignpi/dense/bias/Adam_1save_3/RestoreV2:29*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
?
save_3/Assign_30Assignpi/dense/kernelsave_3/RestoreV2:30*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

:<@*
use_locking(*
validate_shape(
?
save_3/Assign_31Assignpi/dense/kernel/Adamsave_3/RestoreV2:31*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
_output_shapes

:<@
?
save_3/Assign_32Assignpi/dense/kernel/Adam_1save_3/RestoreV2:32*
_output_shapes

:<@*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0
?
save_3/Assign_33Assignpi/dense_1/biassave_3/RestoreV2:33*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:@
?
save_3/Assign_34Assignpi/dense_1/bias/Adamsave_3/RestoreV2:34*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
?
save_3/Assign_35Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:35*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias
?
save_3/Assign_36Assignpi/dense_1/kernelsave_3/RestoreV2:36*
validate_shape(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0
?
save_3/Assign_37Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:37*
_output_shapes

:@@*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
?
save_3/Assign_38Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:38*
validate_shape(*
use_locking(*
_output_shapes

:@@*$
_class
loc:@pi/dense_1/kernel*
T0
?
save_3/Assign_39Assignpi/dense_2/biassave_3/RestoreV2:39*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
?
save_3/Assign_40Assignpi/dense_2/bias/Adamsave_3/RestoreV2:40*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
?
save_3/Assign_41Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:41*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
?
save_3/Assign_42Assignpi/dense_2/kernelsave_3/RestoreV2:42*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
?
save_3/Assign_43Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:43*
T0*
_output_shapes

:@*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
?
save_3/Assign_44Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:44*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@*
use_locking(*
T0
?
save_3/Assign_45Assign
pi/log_stdsave_3/RestoreV2:45*
_class
loc:@pi/log_std*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
?
save_3/Assign_46Assignpi/log_std/Adamsave_3/RestoreV2:46*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(*
T0*
_output_shapes
:
?
save_3/Assign_47Assignpi/log_std/Adam_1save_3/RestoreV2:47*
_class
loc:@pi/log_std*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
?
save_3/Assign_48Assignvc/dense/biassave_3/RestoreV2:48*
T0*
_output_shapes
:@* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(
?
save_3/Assign_49Assignvc/dense/bias/Adamsave_3/RestoreV2:49*
T0* 
_class
loc:@vc/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
:@
?
save_3/Assign_50Assignvc/dense/bias/Adam_1save_3/RestoreV2:50*
validate_shape(* 
_class
loc:@vc/dense/bias*
_output_shapes
:@*
use_locking(*
T0
?
save_3/Assign_51Assignvc/dense/kernelsave_3/RestoreV2:51*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
_output_shapes

:<@*
validate_shape(
?
save_3/Assign_52Assignvc/dense/kernel/Adamsave_3/RestoreV2:52*
_output_shapes

:<@*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0
?
save_3/Assign_53Assignvc/dense/kernel/Adam_1save_3/RestoreV2:53*
validate_shape(*"
_class
loc:@vc/dense/kernel*
use_locking(*
_output_shapes

:<@*
T0
?
save_3/Assign_54Assignvc/dense_1/biassave_3/RestoreV2:54*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes
:@*
validate_shape(
?
save_3/Assign_55Assignvc/dense_1/bias/Adamsave_3/RestoreV2:55*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes
:@*
use_locking(*
validate_shape(
?
save_3/Assign_56Assignvc/dense_1/bias/Adam_1save_3/RestoreV2:56*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
validate_shape(
?
save_3/Assign_57Assignvc/dense_1/kernelsave_3/RestoreV2:57*
_output_shapes

:@@*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(
?
save_3/Assign_58Assignvc/dense_1/kernel/Adamsave_3/RestoreV2:58*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

:@@
?
save_3/Assign_59Assignvc/dense_1/kernel/Adam_1save_3/RestoreV2:59*
validate_shape(*
use_locking(*
_output_shapes

:@@*$
_class
loc:@vc/dense_1/kernel*
T0
?
save_3/Assign_60Assignvc/dense_2/biassave_3/RestoreV2:60*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@vc/dense_2/bias*
use_locking(
?
save_3/Assign_61Assignvc/dense_2/bias/Adamsave_3/RestoreV2:61*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
?
save_3/Assign_62Assignvc/dense_2/bias/Adam_1save_3/RestoreV2:62*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense_2/bias
?
save_3/Assign_63Assignvc/dense_2/kernelsave_3/RestoreV2:63*
validate_shape(*
_output_shapes

:@*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0
?
save_3/Assign_64Assignvc/dense_2/kernel/Adamsave_3/RestoreV2:64*
validate_shape(*
T0*
_output_shapes

:@*$
_class
loc:@vc/dense_2/kernel*
use_locking(
?
save_3/Assign_65Assignvc/dense_2/kernel/Adam_1save_3/RestoreV2:65*
T0*
validate_shape(*
_output_shapes

:@*
use_locking(*$
_class
loc:@vc/dense_2/kernel
?
save_3/Assign_66Assignvf/dense/biassave_3/RestoreV2:66*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias
?
save_3/Assign_67Assignvf/dense/bias/Adamsave_3/RestoreV2:67*
validate_shape(*
T0*
_output_shapes
:@* 
_class
loc:@vf/dense/bias*
use_locking(
?
save_3/Assign_68Assignvf/dense/bias/Adam_1save_3/RestoreV2:68*
validate_shape(*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes
:@
?
save_3/Assign_69Assignvf/dense/kernelsave_3/RestoreV2:69*
validate_shape(*
use_locking(*
_output_shapes

:<@*"
_class
loc:@vf/dense/kernel*
T0
?
save_3/Assign_70Assignvf/dense/kernel/Adamsave_3/RestoreV2:70*
_output_shapes

:<@*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
validate_shape(
?
save_3/Assign_71Assignvf/dense/kernel/Adam_1save_3/RestoreV2:71*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes

:<@*
use_locking(*
T0
?
save_3/Assign_72Assignvf/dense_1/biassave_3/RestoreV2:72*"
_class
loc:@vf/dense_1/bias*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(
?
save_3/Assign_73Assignvf/dense_1/bias/Adamsave_3/RestoreV2:73*
_output_shapes
:@*
validate_shape(*
T0*"
_class
loc:@vf/dense_1/bias*
use_locking(
?
save_3/Assign_74Assignvf/dense_1/bias/Adam_1save_3/RestoreV2:74*
_output_shapes
:@*"
_class
loc:@vf/dense_1/bias*
T0*
validate_shape(*
use_locking(
?
save_3/Assign_75Assignvf/dense_1/kernelsave_3/RestoreV2:75*
use_locking(*
_output_shapes

:@@*
validate_shape(*
T0*$
_class
loc:@vf/dense_1/kernel
?
save_3/Assign_76Assignvf/dense_1/kernel/Adamsave_3/RestoreV2:76*
_output_shapes

:@@*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
use_locking(
?
save_3/Assign_77Assignvf/dense_1/kernel/Adam_1save_3/RestoreV2:77*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel*
_output_shapes

:@@*
T0
?
save_3/Assign_78Assignvf/dense_2/biassave_3/RestoreV2:78*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
_output_shapes
:
?
save_3/Assign_79Assignvf/dense_2/bias/Adamsave_3/RestoreV2:79*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_2/bias
?
save_3/Assign_80Assignvf/dense_2/bias/Adam_1save_3/RestoreV2:80*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
?
save_3/Assign_81Assignvf/dense_2/kernelsave_3/RestoreV2:81*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:@*
validate_shape(
?
save_3/Assign_82Assignvf/dense_2/kernel/Adamsave_3/RestoreV2:82*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@
?
save_3/Assign_83Assignvf/dense_2/kernel/Adam_1save_3/RestoreV2:83*
validate_shape(*
_output_shapes

:@*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel
?
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_51^save_3/Assign_52^save_3/Assign_53^save_3/Assign_54^save_3/Assign_55^save_3/Assign_56^save_3/Assign_57^save_3/Assign_58^save_3/Assign_59^save_3/Assign_6^save_3/Assign_60^save_3/Assign_61^save_3/Assign_62^save_3/Assign_63^save_3/Assign_64^save_3/Assign_65^save_3/Assign_66^save_3/Assign_67^save_3/Assign_68^save_3/Assign_69^save_3/Assign_7^save_3/Assign_70^save_3/Assign_71^save_3/Assign_72^save_3/Assign_73^save_3/Assign_74^save_3/Assign_75^save_3/Assign_76^save_3/Assign_77^save_3/Assign_78^save_3/Assign_79^save_3/Assign_8^save_3/Assign_80^save_3/Assign_81^save_3/Assign_82^save_3/Assign_83^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard "B
save_3/Const:0save_3/Identity:0save_3/restore_all (5 @F8"?P
	variables?O?O
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08
w
lam/dense/kernel:0lam/dense/kernel/Assignlam/dense/kernel/read:02-lam/dense/kernel/Initializer/random_uniform:08
f
lam/dense/bias:0lam/dense/bias/Assignlam/dense/bias/read:02"lam/dense/bias/Initializer/zeros:08

lam/dense_1/kernel:0lam/dense_1/kernel/Assignlam/dense_1/kernel/read:02/lam/dense_1/kernel/Initializer/random_uniform:08
n
lam/dense_1/bias:0lam/dense_1/bias/Assignlam/dense_1/bias/read:02$lam/dense_1/bias/Initializer/zeros:08

lam/dense_2/kernel:0lam/dense_2/kernel/Assignlam/dense_2/kernel/read:02/lam/dense_2/kernel/Initializer/random_uniform:08
n
lam/dense_2/bias:0lam/dense_2/bias/Assignlam/dense_2/bias/read:02$lam/dense_2/bias/Initializer/Const:08
H

Variable:0Variable/AssignVariable/read:02Variable/initial_value:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0
?
pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0
?
pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0
?
pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0
?
pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0
?
pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0
?
pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0
?
pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
h
pi/log_std/Adam:0pi/log_std/Adam/Assignpi/log_std/Adam/read:02#pi/log_std/Adam/Initializer/zeros:0
p
pi/log_std/Adam_1:0pi/log_std/Adam_1/Assignpi/log_std/Adam_1/read:02%pi/log_std/Adam_1/Initializer/zeros:0
P
Variable_1:0Variable_1/AssignVariable_1/read:02Variable_1/initial_value:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
?
lam/dense/kernel/Adam:0lam/dense/kernel/Adam/Assignlam/dense/kernel/Adam/read:02)lam/dense/kernel/Adam/Initializer/zeros:0
?
lam/dense/kernel/Adam_1:0lam/dense/kernel/Adam_1/Assignlam/dense/kernel/Adam_1/read:02+lam/dense/kernel/Adam_1/Initializer/zeros:0
x
lam/dense/bias/Adam:0lam/dense/bias/Adam/Assignlam/dense/bias/Adam/read:02'lam/dense/bias/Adam/Initializer/zeros:0
?
lam/dense/bias/Adam_1:0lam/dense/bias/Adam_1/Assignlam/dense/bias/Adam_1/read:02)lam/dense/bias/Adam_1/Initializer/zeros:0
?
lam/dense_1/kernel/Adam:0lam/dense_1/kernel/Adam/Assignlam/dense_1/kernel/Adam/read:02+lam/dense_1/kernel/Adam/Initializer/zeros:0
?
lam/dense_1/kernel/Adam_1:0 lam/dense_1/kernel/Adam_1/Assign lam/dense_1/kernel/Adam_1/read:02-lam/dense_1/kernel/Adam_1/Initializer/zeros:0
?
lam/dense_1/bias/Adam:0lam/dense_1/bias/Adam/Assignlam/dense_1/bias/Adam/read:02)lam/dense_1/bias/Adam/Initializer/zeros:0
?
lam/dense_1/bias/Adam_1:0lam/dense_1/bias/Adam_1/Assignlam/dense_1/bias/Adam_1/read:02+lam/dense_1/bias/Adam_1/Initializer/zeros:0
?
lam/dense_2/kernel/Adam:0lam/dense_2/kernel/Adam/Assignlam/dense_2/kernel/Adam/read:02+lam/dense_2/kernel/Adam/Initializer/zeros:0
?
lam/dense_2/kernel/Adam_1:0 lam/dense_2/kernel/Adam_1/Assign lam/dense_2/kernel/Adam_1/read:02-lam/dense_2/kernel/Adam_1/Initializer/zeros:0
?
lam/dense_2/bias/Adam:0lam/dense_2/bias/Adam/Assignlam/dense_2/bias/Adam/read:02)lam/dense_2/bias/Adam/Initializer/zeros:0
?
lam/dense_2/bias/Adam_1:0lam/dense_2/bias/Adam_1/Assignlam/dense_2/bias/Adam_1/read:02+lam/dense_2/bias/Adam_1/Initializer/zeros:0
P
Variable_2:0Variable_2/AssignVariable_2/read:02Variable_2/initial_value:0
\
beta1_power_2:0beta1_power_2/Assignbeta1_power_2/read:02beta1_power_2/initial_value:0
\
beta2_power_2:0beta2_power_2/Assignbeta2_power_2/read:02beta2_power_2/initial_value:0
|
vf/dense/kernel/Adam:0vf/dense/kernel/Adam/Assignvf/dense/kernel/Adam/read:02(vf/dense/kernel/Adam/Initializer/zeros:0
?
vf/dense/kernel/Adam_1:0vf/dense/kernel/Adam_1/Assignvf/dense/kernel/Adam_1/read:02*vf/dense/kernel/Adam_1/Initializer/zeros:0
t
vf/dense/bias/Adam:0vf/dense/bias/Adam/Assignvf/dense/bias/Adam/read:02&vf/dense/bias/Adam/Initializer/zeros:0
|
vf/dense/bias/Adam_1:0vf/dense/bias/Adam_1/Assignvf/dense/bias/Adam_1/read:02(vf/dense/bias/Adam_1/Initializer/zeros:0
?
vf/dense_1/kernel/Adam:0vf/dense_1/kernel/Adam/Assignvf/dense_1/kernel/Adam/read:02*vf/dense_1/kernel/Adam/Initializer/zeros:0
?
vf/dense_1/kernel/Adam_1:0vf/dense_1/kernel/Adam_1/Assignvf/dense_1/kernel/Adam_1/read:02,vf/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_1/bias/Adam:0vf/dense_1/bias/Adam/Assignvf/dense_1/bias/Adam/read:02(vf/dense_1/bias/Adam/Initializer/zeros:0
?
vf/dense_1/bias/Adam_1:0vf/dense_1/bias/Adam_1/Assignvf/dense_1/bias/Adam_1/read:02*vf/dense_1/bias/Adam_1/Initializer/zeros:0
?
vf/dense_2/kernel/Adam:0vf/dense_2/kernel/Adam/Assignvf/dense_2/kernel/Adam/read:02*vf/dense_2/kernel/Adam/Initializer/zeros:0
?
vf/dense_2/kernel/Adam_1:0vf/dense_2/kernel/Adam_1/Assignvf/dense_2/kernel/Adam_1/read:02,vf/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_2/bias/Adam:0vf/dense_2/bias/Adam/Assignvf/dense_2/bias/Adam/read:02(vf/dense_2/bias/Adam/Initializer/zeros:0
?
vf/dense_2/bias/Adam_1:0vf/dense_2/bias/Adam_1/Assignvf/dense_2/bias/Adam_1/read:02*vf/dense_2/bias/Adam_1/Initializer/zeros:0
|
vc/dense/kernel/Adam:0vc/dense/kernel/Adam/Assignvc/dense/kernel/Adam/read:02(vc/dense/kernel/Adam/Initializer/zeros:0
?
vc/dense/kernel/Adam_1:0vc/dense/kernel/Adam_1/Assignvc/dense/kernel/Adam_1/read:02*vc/dense/kernel/Adam_1/Initializer/zeros:0
t
vc/dense/bias/Adam:0vc/dense/bias/Adam/Assignvc/dense/bias/Adam/read:02&vc/dense/bias/Adam/Initializer/zeros:0
|
vc/dense/bias/Adam_1:0vc/dense/bias/Adam_1/Assignvc/dense/bias/Adam_1/read:02(vc/dense/bias/Adam_1/Initializer/zeros:0
?
vc/dense_1/kernel/Adam:0vc/dense_1/kernel/Adam/Assignvc/dense_1/kernel/Adam/read:02*vc/dense_1/kernel/Adam/Initializer/zeros:0
?
vc/dense_1/kernel/Adam_1:0vc/dense_1/kernel/Adam_1/Assignvc/dense_1/kernel/Adam_1/read:02,vc/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_1/bias/Adam:0vc/dense_1/bias/Adam/Assignvc/dense_1/bias/Adam/read:02(vc/dense_1/bias/Adam/Initializer/zeros:0
?
vc/dense_1/bias/Adam_1:0vc/dense_1/bias/Adam_1/Assignvc/dense_1/bias/Adam_1/read:02*vc/dense_1/bias/Adam_1/Initializer/zeros:0
?
vc/dense_2/kernel/Adam:0vc/dense_2/kernel/Adam/Assignvc/dense_2/kernel/Adam/read:02*vc/dense_2/kernel/Adam/Initializer/zeros:0
?
vc/dense_2/kernel/Adam_1:0vc/dense_2/kernel/Adam_1/Assignvc/dense_2/kernel/Adam_1/read:02,vc/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_2/bias/Adam:0vc/dense_2/bias/Adam/Assignvc/dense_2/bias/Adam/read:02(vc/dense_2/bias/Adam/Initializer/zeros:0
?
vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0"$
train_op

Adam
Adam_1
Adam_2"?
trainable_variables??
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08
w
lam/dense/kernel:0lam/dense/kernel/Assignlam/dense/kernel/read:02-lam/dense/kernel/Initializer/random_uniform:08
f
lam/dense/bias:0lam/dense/bias/Assignlam/dense/bias/read:02"lam/dense/bias/Initializer/zeros:08

lam/dense_1/kernel:0lam/dense_1/kernel/Assignlam/dense_1/kernel/read:02/lam/dense_1/kernel/Initializer/random_uniform:08
n
lam/dense_1/bias:0lam/dense_1/bias/Assignlam/dense_1/bias/read:02$lam/dense_1/bias/Initializer/zeros:08

lam/dense_2/kernel:0lam/dense_2/kernel/Assignlam/dense_2/kernel/read:02/lam/dense_2/kernel/Initializer/random_uniform:08
n
lam/dense_2/bias:0lam/dense_2/bias/Assignlam/dense_2/bias/read:02$lam/dense_2/bias/Initializer/Const:08*?
serving_default?
)
x$
Placeholder:0?????????<$
v
vf/Squeeze:0?????????%
vc
vc/Squeeze:0?????????.

multiplier 
lam/Squeeze:0?????????%
pi
pi/add:0?????????tensorflow/serving/predict