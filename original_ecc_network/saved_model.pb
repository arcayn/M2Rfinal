¦·
î
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.22v2.6.1-9-gc2363d6d0258

net/ecc_conv/root_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namenet/ecc_conv/root_kernel

,net/ecc_conv/root_kernel/Read/ReadVariableOpReadVariableOpnet/ecc_conv/root_kernel*
_output_shapes

: *
dtype0
z
net/ecc_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namenet/ecc_conv/bias
s
%net/ecc_conv/bias/Read/ReadVariableOpReadVariableOpnet/ecc_conv/bias*
_output_shapes
: *
dtype0

net/ecc_conv_1/root_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_namenet/ecc_conv_1/root_kernel

.net/ecc_conv_1/root_kernel/Read/ReadVariableOpReadVariableOpnet/ecc_conv_1/root_kernel*
_output_shapes

:  *
dtype0
~
net/ecc_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namenet/ecc_conv_1/bias
w
'net/ecc_conv_1/bias/Read/ReadVariableOpReadVariableOpnet/ecc_conv_1/bias*
_output_shapes
: *
dtype0
}
net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *!
shared_namenet/dense/kernel
v
$net/dense/kernel/Read/ReadVariableOpReadVariableOpnet/dense/kernel*
_output_shapes
:	 *
dtype0
u
net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namenet/dense/bias
n
"net/dense/bias/Read/ReadVariableOpReadVariableOpnet/dense/bias*
_output_shapes	
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

net/ecc_conv/FGN_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *,
shared_namenet/ecc_conv/FGN_out/kernel

/net/ecc_conv/FGN_out/kernel/Read/ReadVariableOpReadVariableOpnet/ecc_conv/FGN_out/kernel*
_output_shapes
:	 *
dtype0

net/ecc_conv/FGN_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namenet/ecc_conv/FGN_out/bias

-net/ecc_conv/FGN_out/bias/Read/ReadVariableOpReadVariableOpnet/ecc_conv/FGN_out/bias*
_output_shapes	
: *
dtype0

net/ecc_conv_1/FGN_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_namenet/ecc_conv_1/FGN_out/kernel

1net/ecc_conv_1/FGN_out/kernel/Read/ReadVariableOpReadVariableOpnet/ecc_conv_1/FGN_out/kernel*
_output_shapes
:	*
dtype0

net/ecc_conv_1/FGN_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namenet/ecc_conv_1/FGN_out/bias

/net/ecc_conv_1/FGN_out/bias/Read/ReadVariableOpReadVariableOpnet/ecc_conv_1/FGN_out/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/net/ecc_conv/root_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/net/ecc_conv/root_kernel/m

3Adam/net/ecc_conv/root_kernel/m/Read/ReadVariableOpReadVariableOpAdam/net/ecc_conv/root_kernel/m*
_output_shapes

: *
dtype0

Adam/net/ecc_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/net/ecc_conv/bias/m

,Adam/net/ecc_conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/net/ecc_conv/bias/m*
_output_shapes
: *
dtype0

!Adam/net/ecc_conv_1/root_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!Adam/net/ecc_conv_1/root_kernel/m

5Adam/net/ecc_conv_1/root_kernel/m/Read/ReadVariableOpReadVariableOp!Adam/net/ecc_conv_1/root_kernel/m*
_output_shapes

:  *
dtype0

Adam/net/ecc_conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/net/ecc_conv_1/bias/m

.Adam/net/ecc_conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/net/ecc_conv_1/bias/m*
_output_shapes
: *
dtype0

Adam/net/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *(
shared_nameAdam/net/dense/kernel/m

+Adam/net/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/net/dense/kernel/m*
_output_shapes
:	 *
dtype0

Adam/net/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/net/dense/bias/m
|
)Adam/net/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/net/dense/bias/m*
_output_shapes	
:*
dtype0
¡
"Adam/net/ecc_conv/FGN_out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *3
shared_name$"Adam/net/ecc_conv/FGN_out/kernel/m

6Adam/net/ecc_conv/FGN_out/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/net/ecc_conv/FGN_out/kernel/m*
_output_shapes
:	 *
dtype0

 Adam/net/ecc_conv/FGN_out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/net/ecc_conv/FGN_out/bias/m

4Adam/net/ecc_conv/FGN_out/bias/m/Read/ReadVariableOpReadVariableOp Adam/net/ecc_conv/FGN_out/bias/m*
_output_shapes	
: *
dtype0
¥
$Adam/net/ecc_conv_1/FGN_out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$Adam/net/ecc_conv_1/FGN_out/kernel/m

8Adam/net/ecc_conv_1/FGN_out/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/net/ecc_conv_1/FGN_out/kernel/m*
_output_shapes
:	*
dtype0

"Adam/net/ecc_conv_1/FGN_out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/net/ecc_conv_1/FGN_out/bias/m

6Adam/net/ecc_conv_1/FGN_out/bias/m/Read/ReadVariableOpReadVariableOp"Adam/net/ecc_conv_1/FGN_out/bias/m*
_output_shapes	
:*
dtype0

Adam/net/ecc_conv/root_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/net/ecc_conv/root_kernel/v

3Adam/net/ecc_conv/root_kernel/v/Read/ReadVariableOpReadVariableOpAdam/net/ecc_conv/root_kernel/v*
_output_shapes

: *
dtype0

Adam/net/ecc_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/net/ecc_conv/bias/v

,Adam/net/ecc_conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/net/ecc_conv/bias/v*
_output_shapes
: *
dtype0

!Adam/net/ecc_conv_1/root_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!Adam/net/ecc_conv_1/root_kernel/v

5Adam/net/ecc_conv_1/root_kernel/v/Read/ReadVariableOpReadVariableOp!Adam/net/ecc_conv_1/root_kernel/v*
_output_shapes

:  *
dtype0

Adam/net/ecc_conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/net/ecc_conv_1/bias/v

.Adam/net/ecc_conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/net/ecc_conv_1/bias/v*
_output_shapes
: *
dtype0

Adam/net/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *(
shared_nameAdam/net/dense/kernel/v

+Adam/net/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/net/dense/kernel/v*
_output_shapes
:	 *
dtype0

Adam/net/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/net/dense/bias/v
|
)Adam/net/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/net/dense/bias/v*
_output_shapes	
:*
dtype0
¡
"Adam/net/ecc_conv/FGN_out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *3
shared_name$"Adam/net/ecc_conv/FGN_out/kernel/v

6Adam/net/ecc_conv/FGN_out/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/net/ecc_conv/FGN_out/kernel/v*
_output_shapes
:	 *
dtype0

 Adam/net/ecc_conv/FGN_out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/net/ecc_conv/FGN_out/bias/v

4Adam/net/ecc_conv/FGN_out/bias/v/Read/ReadVariableOpReadVariableOp Adam/net/ecc_conv/FGN_out/bias/v*
_output_shapes	
: *
dtype0
¥
$Adam/net/ecc_conv_1/FGN_out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$Adam/net/ecc_conv_1/FGN_out/kernel/v

8Adam/net/ecc_conv_1/FGN_out/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/net/ecc_conv_1/FGN_out/kernel/v*
_output_shapes
:	*
dtype0

"Adam/net/ecc_conv_1/FGN_out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/net/ecc_conv_1/FGN_out/bias/v

6Adam/net/ecc_conv_1/FGN_out/bias/v/Read/ReadVariableOpReadVariableOp"Adam/net/ecc_conv_1/FGN_out/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
â9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*9
value9B9 B9
Î
masking
	conv1
	conv2
global_pool1
global_pool2
	activ
	dense
	optimizer
	trainable_variables

	variables
regularization_losses
	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api

kwargs_keys
kernel_network_layers
root_kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

kwargs_keys
kernel_network_layers
root_kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
R
"trainable_variables
#	variables
$regularization_losses
%	keras_api

&	keras_api

'	keras_api
h

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
ö
.iter

/beta_1

0beta_2
	1decay
2learning_ratemnmompmq(mr)ms3mt4mu5mv6mwvxvyvzv{(v|)v}3v~4v5v6v
F
0
1
32
43
4
5
56
67
(8
)9
F
0
1
32
43
4
5
56
67
(8
)9
 
­
	trainable_variables
7layer_metrics
8metrics

	variables
9layer_regularization_losses
regularization_losses

:layers
;non_trainable_variables
 
 
 
 
­
trainable_variables
<layer_metrics
=metrics
>layer_regularization_losses
	variables
regularization_losses

?layers
@non_trainable_variables
 

A0
ZX
VARIABLE_VALUEnet/ecc_conv/root_kernel,conv1/root_kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEnet/ecc_conv/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
32
43

0
1
32
43
 
­
trainable_variables
Blayer_metrics
Cmetrics
Dlayer_regularization_losses
	variables
regularization_losses

Elayers
Fnon_trainable_variables
 

G0
\Z
VARIABLE_VALUEnet/ecc_conv_1/root_kernel,conv2/root_kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEnet/ecc_conv_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
52
63

0
1
52
63
 
­
trainable_variables
Hlayer_metrics
Imetrics
Jlayer_regularization_losses
	variables
 regularization_losses

Klayers
Lnon_trainable_variables
 
 
 
­
"trainable_variables
Mlayer_metrics
Nmetrics
Olayer_regularization_losses
#	variables
$regularization_losses

Players
Qnon_trainable_variables
 
 
MK
VARIABLE_VALUEnet/dense/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEnet/dense/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
­
*trainable_variables
Rlayer_metrics
Smetrics
Tlayer_regularization_losses
+	variables
,regularization_losses

Ulayers
Vnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEnet/ecc_conv/FGN_out/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEnet/ecc_conv/FGN_out/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEnet/ecc_conv_1/FGN_out/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEnet/ecc_conv_1/FGN_out/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
 

W0
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
h

3kernel
4bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
 
 
 

A0
 
h

5kernel
6bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
 
 
 

G0
 
 
 
 
 
 
 
 
 
 
 
4
	`total
	acount
b	variables
c	keras_api

30
41

30
41
 
­
Xtrainable_variables
dlayer_metrics
emetrics
flayer_regularization_losses
Y	variables
Zregularization_losses

glayers
hnon_trainable_variables

50
61

50
61
 
­
\trainable_variables
ilayer_metrics
jmetrics
klayer_regularization_losses
]	variables
^regularization_losses

llayers
mnon_trainable_variables
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

b	variables
 
 
 
 
 
 
 
 
 
 
}{
VARIABLE_VALUEAdam/net/ecc_conv/root_kernel/mHconv1/root_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/net/ecc_conv/bias/mAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/net/ecc_conv_1/root_kernel/mHconv2/root_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/net/ecc_conv_1/bias/mAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/net/dense/kernel/mCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/net/dense/bias/mAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/net/ecc_conv/FGN_out/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/net/ecc_conv/FGN_out/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/net/ecc_conv_1/FGN_out/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/net/ecc_conv_1/FGN_out/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/net/ecc_conv/root_kernel/vHconv1/root_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/net/ecc_conv/bias/vAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/net/ecc_conv_1/root_kernel/vHconv2/root_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/net/ecc_conv_1/bias/vAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/net/dense/kernel/vCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/net/dense/bias/vAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/net/ecc_conv/FGN_out/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/net/ecc_conv/FGN_out/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/net/ecc_conv_1/FGN_out/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/net/ecc_conv_1/FGN_out/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ


serving_default_input_2Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ



serving_default_input_3Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
dtype0	*$
shape:ÿÿÿÿÿÿÿÿÿ


ï
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3net/ecc_conv/FGN_out/kernelnet/ecc_conv/FGN_out/biasnet/ecc_conv/root_kernelnet/ecc_conv/biasnet/ecc_conv_1/FGN_out/kernelnet/ecc_conv_1/FGN_out/biasnet/ecc_conv_1/root_kernelnet/ecc_conv_1/biasnet/dense/kernelnet/dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_150839
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ß
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,net/ecc_conv/root_kernel/Read/ReadVariableOp%net/ecc_conv/bias/Read/ReadVariableOp.net/ecc_conv_1/root_kernel/Read/ReadVariableOp'net/ecc_conv_1/bias/Read/ReadVariableOp$net/dense/kernel/Read/ReadVariableOp"net/dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/net/ecc_conv/FGN_out/kernel/Read/ReadVariableOp-net/ecc_conv/FGN_out/bias/Read/ReadVariableOp1net/ecc_conv_1/FGN_out/kernel/Read/ReadVariableOp/net/ecc_conv_1/FGN_out/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/net/ecc_conv/root_kernel/m/Read/ReadVariableOp,Adam/net/ecc_conv/bias/m/Read/ReadVariableOp5Adam/net/ecc_conv_1/root_kernel/m/Read/ReadVariableOp.Adam/net/ecc_conv_1/bias/m/Read/ReadVariableOp+Adam/net/dense/kernel/m/Read/ReadVariableOp)Adam/net/dense/bias/m/Read/ReadVariableOp6Adam/net/ecc_conv/FGN_out/kernel/m/Read/ReadVariableOp4Adam/net/ecc_conv/FGN_out/bias/m/Read/ReadVariableOp8Adam/net/ecc_conv_1/FGN_out/kernel/m/Read/ReadVariableOp6Adam/net/ecc_conv_1/FGN_out/bias/m/Read/ReadVariableOp3Adam/net/ecc_conv/root_kernel/v/Read/ReadVariableOp,Adam/net/ecc_conv/bias/v/Read/ReadVariableOp5Adam/net/ecc_conv_1/root_kernel/v/Read/ReadVariableOp.Adam/net/ecc_conv_1/bias/v/Read/ReadVariableOp+Adam/net/dense/kernel/v/Read/ReadVariableOp)Adam/net/dense/bias/v/Read/ReadVariableOp6Adam/net/ecc_conv/FGN_out/kernel/v/Read/ReadVariableOp4Adam/net/ecc_conv/FGN_out/bias/v/Read/ReadVariableOp8Adam/net/ecc_conv_1/FGN_out/kernel/v/Read/ReadVariableOp6Adam/net/ecc_conv_1/FGN_out/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_152013
ö	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenet/ecc_conv/root_kernelnet/ecc_conv/biasnet/ecc_conv_1/root_kernelnet/ecc_conv_1/biasnet/dense/kernelnet/dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratenet/ecc_conv/FGN_out/kernelnet/ecc_conv/FGN_out/biasnet/ecc_conv_1/FGN_out/kernelnet/ecc_conv_1/FGN_out/biastotalcountAdam/net/ecc_conv/root_kernel/mAdam/net/ecc_conv/bias/m!Adam/net/ecc_conv_1/root_kernel/mAdam/net/ecc_conv_1/bias/mAdam/net/dense/kernel/mAdam/net/dense/bias/m"Adam/net/ecc_conv/FGN_out/kernel/m Adam/net/ecc_conv/FGN_out/bias/m$Adam/net/ecc_conv_1/FGN_out/kernel/m"Adam/net/ecc_conv_1/FGN_out/bias/mAdam/net/ecc_conv/root_kernel/vAdam/net/ecc_conv/bias/v!Adam/net/ecc_conv_1/root_kernel/vAdam/net/ecc_conv_1/bias/vAdam/net/dense/kernel/vAdam/net/dense/bias/v"Adam/net/ecc_conv/FGN_out/kernel/v Adam/net/ecc_conv/FGN_out/bias/v$Adam/net/ecc_conv_1/FGN_out/kernel/v"Adam/net/ecc_conv_1/FGN_out/bias/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_152134Þ
ªæ
å
?__inference_net_layer_call_and_return_conditional_losses_151627
input_1
input_2
input_3	E
2ecc_conv_fgn_out_tensordot_readvariableop_resource:	 ?
0ecc_conv_fgn_out_biasadd_readvariableop_resource:	 :
(ecc_conv_shape_3_readvariableop_resource: 6
(ecc_conv_biasadd_readvariableop_resource: G
4ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	A
2ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	<
*ecc_conv_1_shape_3_readvariableop_resource:  8
*ecc_conv_1_biasadd_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	 4
%dense_biasadd_readvariableop_resource:	
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢ecc_conv/BiasAdd/ReadVariableOp¢'ecc_conv/FGN_out/BiasAdd/ReadVariableOp¢)ecc_conv/FGN_out/Tensordot/ReadVariableOp¢!ecc_conv/transpose/ReadVariableOp¢!ecc_conv_1/BiasAdd/ReadVariableOp¢)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp¢+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp¢#ecc_conv_1/transpose/ReadVariableOp
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!graph_masking/strided_slice/stack
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice/stack_1
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#graph_masking/strided_slice/stack_2Å
graph_masking/strided_sliceStridedSliceinput_1*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
graph_masking/strided_slice
#graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice_1/stack
%graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%graph_masking/strided_slice_1/stack_1
%graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%graph_masking/strided_slice_1/stack_2Í
graph_masking/strided_slice_1StridedSliceinput_1,graph_masking/strided_slice_1/stack:output:0.graph_masking/strided_slice_1/stack_1:output:0.graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ellipsis_mask*
end_mask2
graph_masking/strided_slice_1t
ecc_conv/ShapeShape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
ecc_conv/strided_slice/stack
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice/stack_1
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slicex
ecc_conv/Shape_1Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_1
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice_1/stack
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/strided_slice_1/stack_1
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv/strided_slice_1/stack_2¤
ecc_conv/strided_slice_1StridedSliceecc_conv/Shape_1:output:0'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice_1
ecc_conv/FGN_out/CastCastinput_3*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv/FGN_out/CastÊ
)ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp2ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype02+
)ecc_conv/FGN_out/Tensordot/ReadVariableOp
ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
ecc_conv/FGN_out/Tensordot/axes
ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
ecc_conv/FGN_out/Tensordot/free
 ecc_conv/FGN_out/Tensordot/ShapeShapeecc_conv/FGN_out/Cast:y:0*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/Shape
(ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/GatherV2/axis¦
#ecc_conv/FGN_out/Tensordot/GatherV2GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/free:output:01ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/GatherV2
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axis¬
%ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/axes:output:03ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv/FGN_out/Tensordot/GatherV2_1
 ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/FGN_out/Tensordot/ConstÄ
ecc_conv/FGN_out/Tensordot/ProdProd,ecc_conv/FGN_out/Tensordot/GatherV2:output:0)ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
ecc_conv/FGN_out/Tensordot/Prod
"ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_1Ì
!ecc_conv/FGN_out/Tensordot/Prod_1Prod.ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0+ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!ecc_conv/FGN_out/Tensordot/Prod_1
&ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&ecc_conv/FGN_out/Tensordot/concat/axis
!ecc_conv/FGN_out/Tensordot/concatConcatV2(ecc_conv/FGN_out/Tensordot/free:output:0(ecc_conv/FGN_out/Tensordot/axes:output:0/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!ecc_conv/FGN_out/Tensordot/concatÐ
 ecc_conv/FGN_out/Tensordot/stackPack(ecc_conv/FGN_out/Tensordot/Prod:output:0*ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/stackÚ
$ecc_conv/FGN_out/Tensordot/transpose	Transposeecc_conv/FGN_out/Cast:y:0*ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2&
$ecc_conv/FGN_out/Tensordot/transposeã
"ecc_conv/FGN_out/Tensordot/ReshapeReshape(ecc_conv/FGN_out/Tensordot/transpose:y:0)ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"ecc_conv/FGN_out/Tensordot/Reshapeã
!ecc_conv/FGN_out/Tensordot/MatMulMatMul+ecc_conv/FGN_out/Tensordot/Reshape:output:01ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ecc_conv/FGN_out/Tensordot/MatMul
"ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_2
(ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/concat_1/axis
#ecc_conv/FGN_out/Tensordot/concat_1ConcatV2,ecc_conv/FGN_out/Tensordot/GatherV2:output:0+ecc_conv/FGN_out/Tensordot/Const_2:output:01ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/concat_1Ù
ecc_conv/FGN_out/TensordotReshape+ecc_conv/FGN_out/Tensordot/MatMul:product:0,ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/TensordotÀ
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpÐ
ecc_conv/FGN_out/BiasAddBiasAdd#ecc_conv/FGN_out/Tensordot:output:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/BiasAdd
ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape/shape/0v
ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape/shape/3
ecc_conv/Reshape/shapePack!ecc_conv/Reshape/shape/0:output:0ecc_conv/strided_slice:output:0ecc_conv/strided_slice:output:0!ecc_conv/Reshape/shape/3:output:0!ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape/shapeÌ
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_2/stack
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_2/stack_1
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_2/stack_2Á
ecc_conv/strided_slice_2StridedSliceinput_2'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv/strided_slice_2¨
ecc_conv/mulMulecc_conv/Reshape:output:0!ecc_conv/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv/mulÏ
ecc_conv/einsum/EinsumEinsumecc_conv/mul:z:0$graph_masking/strided_slice:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv/einsum/Einsumx
ecc_conv/Shape_2Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_2y
ecc_conv/unstackUnpackecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv/unstack«
ecc_conv/Shape_3/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02!
ecc_conv/Shape_3/ReadVariableOpu
ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/Shape_3{
ecc_conv/unstack_1Unpackecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv/unstack_1
ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
ecc_conv/Reshape_1/shape®
ecc_conv/Reshape_1Reshape$graph_masking/strided_slice:output:0!ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape_1¯
!ecc_conv/transpose/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02#
!ecc_conv/transpose/ReadVariableOp
ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/transpose/perm«
ecc_conv/transpose	Transpose)ecc_conv/transpose/ReadVariableOp:value:0 ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: 2
ecc_conv/transpose
ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
ecc_conv/Reshape_2/shape
ecc_conv/Reshape_2Reshapeecc_conv/transpose:y:0!ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: 2
ecc_conv/Reshape_2
ecc_conv/MatMulMatMulecc_conv/Reshape_1:output:0ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv/MatMulz
ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv/Reshape_3/shape/1z
ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape_3/shape/2Ï
ecc_conv/Reshape_3/shapePackecc_conv/unstack:output:0#ecc_conv/Reshape_3/shape/1:output:0#ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape_3/shape§
ecc_conv/Reshape_3Reshapeecc_conv/MatMul:product:0!ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reshape_3
ecc_conv/addAddV2ecc_conv/einsum/Einsum:output:0ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/add§
ecc_conv/BiasAdd/ReadVariableOpReadVariableOp(ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
ecc_conv/BiasAdd/ReadVariableOp 
ecc_conv/BiasAddBiasAddecc_conv/add:z:0'ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/BiasAdd 
ecc_conv/mul_1Mulecc_conv/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/mul_1p
ecc_conv/ReluReluecc_conv/mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reluo
ecc_conv_1/ShapeShapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape
ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2 
ecc_conv_1/strided_slice/stack
 ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice/stack_1
 ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv_1/strided_slice/stack_2¤
ecc_conv_1/strided_sliceStridedSliceecc_conv_1/Shape:output:0'ecc_conv_1/strided_slice/stack:output:0)ecc_conv_1/strided_slice/stack_1:output:0)ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slices
ecc_conv_1/Shape_1Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_1
 ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice_1/stack
"ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/strided_slice_1/stack_1
"ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ecc_conv_1/strided_slice_1/stack_2°
ecc_conv_1/strided_slice_1StridedSliceecc_conv_1/Shape_1:output:0)ecc_conv_1/strided_slice_1/stack:output:0+ecc_conv_1/strided_slice_1/stack_1:output:0+ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slice_1
ecc_conv_1/FGN_out/CastCastinput_3*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/CastÐ
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02-
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp
!ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!ecc_conv_1/FGN_out/Tensordot/axes
!ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!ecc_conv_1/FGN_out/Tensordot/free
"ecc_conv_1/FGN_out/Tensordot/ShapeShapeecc_conv_1/FGN_out/Cast:y:0*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/Shape
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axis°
%ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/free:output:03ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/GatherV2
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis¶
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:05ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1
"ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/FGN_out/Tensordot/ConstÌ
!ecc_conv_1/FGN_out/Tensordot/ProdProd.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!ecc_conv_1/FGN_out/Tensordot/Prod
$ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$ecc_conv_1/FGN_out/Tensordot/Const_1Ô
#ecc_conv_1/FGN_out/Tensordot/Prod_1Prod0ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#ecc_conv_1/FGN_out/Tensordot/Prod_1
(ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv_1/FGN_out/Tensordot/concat/axis
#ecc_conv_1/FGN_out/Tensordot/concatConcatV2*ecc_conv_1/FGN_out/Tensordot/free:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:01ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv_1/FGN_out/Tensordot/concatØ
"ecc_conv_1/FGN_out/Tensordot/stackPack*ecc_conv_1/FGN_out/Tensordot/Prod:output:0,ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/stackâ
&ecc_conv_1/FGN_out/Tensordot/transpose	Transposeecc_conv_1/FGN_out/Cast:y:0,ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2(
&ecc_conv_1/FGN_out/Tensordot/transposeë
$ecc_conv_1/FGN_out/Tensordot/ReshapeReshape*ecc_conv_1/FGN_out/Tensordot/transpose:y:0+ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$ecc_conv_1/FGN_out/Tensordot/Reshapeë
#ecc_conv_1/FGN_out/Tensordot/MatMulMatMul-ecc_conv_1/FGN_out/Tensordot/Reshape:output:03ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ecc_conv_1/FGN_out/Tensordot/MatMul
$ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$ecc_conv_1/FGN_out/Tensordot/Const_2
*ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/concat_1/axis
%ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_1/FGN_out/Tensordot/Const_2:output:03ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/concat_1á
ecc_conv_1/FGN_out/TensordotReshape-ecc_conv_1/FGN_out/Tensordot/MatMul:product:0.ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/TensordotÆ
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpØ
ecc_conv_1/FGN_out/BiasAddBiasAdd%ecc_conv_1/FGN_out/Tensordot:output:01ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/BiasAdd
ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape/shape/0z
ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape/shape/3
ecc_conv_1/Reshape/shapePack#ecc_conv_1/Reshape/shape/0:output:0!ecc_conv_1/strided_slice:output:0!ecc_conv_1/strided_slice:output:0#ecc_conv_1/Reshape/shape/3:output:0#ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape/shapeÔ
ecc_conv_1/ReshapeReshape#ecc_conv_1/FGN_out/BiasAdd:output:0!ecc_conv_1/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape
 ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv_1/strided_slice_2/stack
"ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2$
"ecc_conv_1/strided_slice_2/stack_1
"ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2$
"ecc_conv_1/strided_slice_2/stack_2Ë
ecc_conv_1/strided_slice_2StridedSliceinput_2)ecc_conv_1/strided_slice_2/stack:output:0+ecc_conv_1/strided_slice_2/stack_1:output:0+ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv_1/strided_slice_2°
ecc_conv_1/mulMulecc_conv_1/Reshape:output:0#ecc_conv_1/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/mulÌ
ecc_conv_1/einsum/EinsumEinsumecc_conv_1/mul:z:0ecc_conv/Relu:activations:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv_1/einsum/Einsums
ecc_conv_1/Shape_2Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_2
ecc_conv_1/unstackUnpackecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv_1/unstack±
!ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02#
!ecc_conv_1/Shape_3/ReadVariableOpy
ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        2
ecc_conv_1/Shape_3
ecc_conv_1/unstack_1Unpackecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv_1/unstack_1
ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
ecc_conv_1/Reshape_1/shape«
ecc_conv_1/Reshape_1Reshapeecc_conv/Relu:activations:0#ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/Reshape_1µ
#ecc_conv_1/transpose/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02%
#ecc_conv_1/transpose/ReadVariableOp
ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv_1/transpose/perm³
ecc_conv_1/transpose	Transpose+ecc_conv_1/transpose/ReadVariableOp:value:0"ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/transpose
ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
ecc_conv_1/Reshape_2/shape
ecc_conv_1/Reshape_2Reshapeecc_conv_1/transpose:y:0#ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/Reshape_2 
ecc_conv_1/MatMulMatMulecc_conv_1/Reshape_1:output:0ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/MatMul~
ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv_1/Reshape_3/shape/1~
ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape_3/shape/2Ù
ecc_conv_1/Reshape_3/shapePackecc_conv_1/unstack:output:0%ecc_conv_1/Reshape_3/shape/1:output:0%ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape_3/shape¯
ecc_conv_1/Reshape_3Reshapeecc_conv_1/MatMul:product:0#ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/Reshape_3¡
ecc_conv_1/addAddV2!ecc_conv_1/einsum/Einsum:output:0ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/add­
!ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!ecc_conv_1/BiasAdd/ReadVariableOp¨
ecc_conv_1/BiasAddBiasAddecc_conv_1/add:z:0)ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/BiasAdd¦
ecc_conv_1/mul_1Mulecc_conv_1/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/mul_1
%global_max_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ2'
%global_max_pool/Max/reduction_indices©
global_max_pool/MaxMaxecc_conv_1/mul_1:z:0.global_max_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
global_max_pool/Max 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulglobal_max_pool/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityË
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp ^ecc_conv/BiasAdd/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp*^ecc_conv/FGN_out/Tensordot/ReadVariableOp"^ecc_conv/transpose/ReadVariableOp"^ecc_conv_1/BiasAdd/ReadVariableOp*^ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_1/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2B
ecc_conv/BiasAdd/ReadVariableOpecc_conv/BiasAdd/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2V
)ecc_conv/FGN_out/Tensordot/ReadVariableOp)ecc_conv/FGN_out/Tensordot/ReadVariableOp2F
!ecc_conv/transpose/ReadVariableOp!ecc_conv/transpose/ReadVariableOp2F
!ecc_conv_1/BiasAdd/ReadVariableOp!ecc_conv_1/BiasAdd/ReadVariableOp2V
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_1/transpose/ReadVariableOp#ecc_conv_1/transpose/ReadVariableOp:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_2:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_3
¡
e
I__inference_graph_masking_layer_call_and_return_conditional_losses_151643

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
$
´
?__inference_net_layer_call_and_return_conditional_losses_150526

inputs
inputs_1
inputs_2	"
ecc_conv_150401:	 
ecc_conv_150403:	 !
ecc_conv_150405: 
ecc_conv_150407: $
ecc_conv_1_150491:	 
ecc_conv_1_150493:	#
ecc_conv_1_150495:  
ecc_conv_1_150497: 
dense_150520:	 
dense_150522:	
identity¢dense/StatefulPartitionedCall¢ ecc_conv/StatefulPartitionedCall¢"ecc_conv_1/StatefulPartitionedCallç
graph_masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_graph_masking_layer_call_and_return_conditional_losses_1503132
graph_masking/PartitionedCall
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2#
!graph_masking/strided_slice/stack
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#graph_masking/strided_slice/stack_1
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#graph_masking/strided_slice/stack_2Â
graph_masking/strided_sliceStridedSliceinputs*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ellipsis_mask*
end_mask2
graph_masking/strided_slice
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCall&graph_masking/PartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_150401ecc_conv_150403ecc_conv_150405ecc_conv_150407*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_ecc_conv_layer_call_and_return_conditional_losses_1504002"
 ecc_conv/StatefulPartitionedCall¬
"ecc_conv_1/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_1_150491ecc_conv_1_150493ecc_conv_1_150495ecc_conv_1_150497*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_ecc_conv_1_layer_call_and_return_conditional_losses_1504902$
"ecc_conv_1/StatefulPartitionedCall
global_max_pool/PartitionedCallPartitionedCall+ecc_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_global_max_pool_layer_call_and_return_conditional_losses_1505062!
global_max_pool/PartitionedCall¨
dense/StatefulPartitionedCallStatefulPartitionedCall(global_max_pool/PartitionedCall:output:0dense_150520dense_150522*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1505192
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¶
NoOpNoOp^dense/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall#^ecc_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2H
"ecc_conv_1/StatefulPartitionedCall"ecc_conv_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
ÙW
ß
F__inference_ecc_conv_1_layer_call_and_return_conditional_losses_151830
inputs_0
inputs_1
inputs_2	

mask_0<
)fgn_out_tensordot_readvariableop_resource:	6
'fgn_out_biasadd_readvariableop_resource:	1
shape_3_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢FGN_out/BiasAdd/ReadVariableOp¢ FGN_out/Tensordot/ReadVariableOp¢transpose/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1w
FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Cast¯
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02"
 FGN_out/Tensordot/ReadVariableOpz
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
FGN_out/Tensordot/axes
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
FGN_out/Tensordot/freer
FGN_out/Tensordot/ShapeShapeFGN_out/Cast:y:0*
T0*
_output_shapes
:2
FGN_out/Tensordot/Shape
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/GatherV2/axisù
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!FGN_out/Tensordot/GatherV2_1/axisÿ
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2_1|
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const 
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const_1¨
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod_1
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
FGN_out/Tensordot/concat/axisØ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat¬
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/stack¶
FGN_out/Tensordot/transpose	TransposeFGN_out/Cast:y:0!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Tensordot/transpose¿
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
FGN_out/Tensordot/Reshape¿
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
FGN_out/Tensordot/MatMul
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
FGN_out/Tensordot/Const_2
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/concat_1/axiså
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat_1µ
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Tensordot¥
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
FGN_out/BiasAdd/ReadVariableOp¬
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/BiasAddm
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3Ò
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape¨
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2	
Reshape
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_2/stack_2
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
strided_slice_2
mulMulReshape:output:0strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
mul
einsum/EinsumEinsummul:z:0inputs_0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
einsum/EinsumJ
Shape_2Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
Reshape_1/shapew
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Reshape_1
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  2
	transposes
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
Reshape_2/shapes
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

:  2
	Reshape_2t
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMulh
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape_3/shape/1h
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_3/shape/2¢
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_3/shape
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
	Reshape_3u
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp|
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2	
BiasAdde
mul_1MulBiasAdd:output:0mask_0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
mul_1h
IdentityIdentity	mul_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

IdentityÆ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ
: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namemask/0
ÁW
Û
F__inference_ecc_conv_1_layer_call_and_return_conditional_losses_150490

inputs
inputs_1
inputs_2	
mask<
)fgn_out_tensordot_readvariableop_resource:	6
'fgn_out_biasadd_readvariableop_resource:	1
shape_3_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢FGN_out/BiasAdd/ReadVariableOp¢ FGN_out/Tensordot/ReadVariableOp¢transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1w
FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Cast¯
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02"
 FGN_out/Tensordot/ReadVariableOpz
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
FGN_out/Tensordot/axes
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
FGN_out/Tensordot/freer
FGN_out/Tensordot/ShapeShapeFGN_out/Cast:y:0*
T0*
_output_shapes
:2
FGN_out/Tensordot/Shape
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/GatherV2/axisù
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!FGN_out/Tensordot/GatherV2_1/axisÿ
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2_1|
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const 
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const_1¨
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod_1
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
FGN_out/Tensordot/concat/axisØ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat¬
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/stack¶
FGN_out/Tensordot/transpose	TransposeFGN_out/Cast:y:0!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Tensordot/transpose¿
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
FGN_out/Tensordot/Reshape¿
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
FGN_out/Tensordot/MatMul
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
FGN_out/Tensordot/Const_2
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/concat_1/axiså
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat_1µ
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Tensordot¥
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
FGN_out/BiasAdd/ReadVariableOp¬
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/BiasAddm
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3Ò
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape¨
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2	
Reshape
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_2/stack_2
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
strided_slice_2
mulMulReshape:output:0strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
mul
einsum/EinsumEinsummul:z:0inputs*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
einsum/EinsumH
Shape_2Shapeinputs*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
Reshape_1/shapeu
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Reshape_1
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:  *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  2
	transposes
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
Reshape_2/shapes
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

:  2
	Reshape_2t
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMulh
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape_3/shape/1h
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_3/shape/2¢
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_3/shape
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
	Reshape_3u
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp|
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2	
BiasAddc
mul_1MulBiasAdd:output:0mask*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
mul_1h
IdentityIdentity	mul_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

IdentityÆ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ
: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs:QM
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namemask
ã

$__inference_net_layer_call_fn_150893
inputs_0
inputs_1
inputs_2	
unknown:	 
	unknown_0:	 
	unknown_1: 
	unknown_2: 
	unknown_3:	
	unknown_4:	
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:	
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_net_layer_call_and_return_conditional_losses_1505262
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2
¡
e
I__inference_graph_masking_layer_call_and_return_conditional_losses_150615

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¶æ
è
?__inference_net_layer_call_and_return_conditional_losses_151287
inputs_0
inputs_1
inputs_2	E
2ecc_conv_fgn_out_tensordot_readvariableop_resource:	 ?
0ecc_conv_fgn_out_biasadd_readvariableop_resource:	 :
(ecc_conv_shape_3_readvariableop_resource: 6
(ecc_conv_biasadd_readvariableop_resource: G
4ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	A
2ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	<
*ecc_conv_1_shape_3_readvariableop_resource:  8
*ecc_conv_1_biasadd_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	 4
%dense_biasadd_readvariableop_resource:	
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢ecc_conv/BiasAdd/ReadVariableOp¢'ecc_conv/FGN_out/BiasAdd/ReadVariableOp¢)ecc_conv/FGN_out/Tensordot/ReadVariableOp¢!ecc_conv/transpose/ReadVariableOp¢!ecc_conv_1/BiasAdd/ReadVariableOp¢)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp¢+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp¢#ecc_conv_1/transpose/ReadVariableOp
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!graph_masking/strided_slice/stack
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice/stack_1
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#graph_masking/strided_slice/stack_2Æ
graph_masking/strided_sliceStridedSliceinputs_0*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
graph_masking/strided_slice
#graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice_1/stack
%graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%graph_masking/strided_slice_1/stack_1
%graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%graph_masking/strided_slice_1/stack_2Î
graph_masking/strided_slice_1StridedSliceinputs_0,graph_masking/strided_slice_1/stack:output:0.graph_masking/strided_slice_1/stack_1:output:0.graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ellipsis_mask*
end_mask2
graph_masking/strided_slice_1t
ecc_conv/ShapeShape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
ecc_conv/strided_slice/stack
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice/stack_1
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slicex
ecc_conv/Shape_1Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_1
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice_1/stack
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/strided_slice_1/stack_1
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv/strided_slice_1/stack_2¤
ecc_conv/strided_slice_1StridedSliceecc_conv/Shape_1:output:0'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice_1
ecc_conv/FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv/FGN_out/CastÊ
)ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp2ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype02+
)ecc_conv/FGN_out/Tensordot/ReadVariableOp
ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
ecc_conv/FGN_out/Tensordot/axes
ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
ecc_conv/FGN_out/Tensordot/free
 ecc_conv/FGN_out/Tensordot/ShapeShapeecc_conv/FGN_out/Cast:y:0*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/Shape
(ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/GatherV2/axis¦
#ecc_conv/FGN_out/Tensordot/GatherV2GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/free:output:01ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/GatherV2
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axis¬
%ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/axes:output:03ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv/FGN_out/Tensordot/GatherV2_1
 ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/FGN_out/Tensordot/ConstÄ
ecc_conv/FGN_out/Tensordot/ProdProd,ecc_conv/FGN_out/Tensordot/GatherV2:output:0)ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
ecc_conv/FGN_out/Tensordot/Prod
"ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_1Ì
!ecc_conv/FGN_out/Tensordot/Prod_1Prod.ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0+ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!ecc_conv/FGN_out/Tensordot/Prod_1
&ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&ecc_conv/FGN_out/Tensordot/concat/axis
!ecc_conv/FGN_out/Tensordot/concatConcatV2(ecc_conv/FGN_out/Tensordot/free:output:0(ecc_conv/FGN_out/Tensordot/axes:output:0/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!ecc_conv/FGN_out/Tensordot/concatÐ
 ecc_conv/FGN_out/Tensordot/stackPack(ecc_conv/FGN_out/Tensordot/Prod:output:0*ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/stackÚ
$ecc_conv/FGN_out/Tensordot/transpose	Transposeecc_conv/FGN_out/Cast:y:0*ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2&
$ecc_conv/FGN_out/Tensordot/transposeã
"ecc_conv/FGN_out/Tensordot/ReshapeReshape(ecc_conv/FGN_out/Tensordot/transpose:y:0)ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"ecc_conv/FGN_out/Tensordot/Reshapeã
!ecc_conv/FGN_out/Tensordot/MatMulMatMul+ecc_conv/FGN_out/Tensordot/Reshape:output:01ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ecc_conv/FGN_out/Tensordot/MatMul
"ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_2
(ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/concat_1/axis
#ecc_conv/FGN_out/Tensordot/concat_1ConcatV2,ecc_conv/FGN_out/Tensordot/GatherV2:output:0+ecc_conv/FGN_out/Tensordot/Const_2:output:01ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/concat_1Ù
ecc_conv/FGN_out/TensordotReshape+ecc_conv/FGN_out/Tensordot/MatMul:product:0,ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/TensordotÀ
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpÐ
ecc_conv/FGN_out/BiasAddBiasAdd#ecc_conv/FGN_out/Tensordot:output:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/BiasAdd
ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape/shape/0v
ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape/shape/3
ecc_conv/Reshape/shapePack!ecc_conv/Reshape/shape/0:output:0ecc_conv/strided_slice:output:0ecc_conv/strided_slice:output:0!ecc_conv/Reshape/shape/3:output:0!ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape/shapeÌ
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_2/stack
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_2/stack_1
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_2/stack_2Â
ecc_conv/strided_slice_2StridedSliceinputs_1'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv/strided_slice_2¨
ecc_conv/mulMulecc_conv/Reshape:output:0!ecc_conv/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv/mulÏ
ecc_conv/einsum/EinsumEinsumecc_conv/mul:z:0$graph_masking/strided_slice:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv/einsum/Einsumx
ecc_conv/Shape_2Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_2y
ecc_conv/unstackUnpackecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv/unstack«
ecc_conv/Shape_3/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02!
ecc_conv/Shape_3/ReadVariableOpu
ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/Shape_3{
ecc_conv/unstack_1Unpackecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv/unstack_1
ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
ecc_conv/Reshape_1/shape®
ecc_conv/Reshape_1Reshape$graph_masking/strided_slice:output:0!ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape_1¯
!ecc_conv/transpose/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02#
!ecc_conv/transpose/ReadVariableOp
ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/transpose/perm«
ecc_conv/transpose	Transpose)ecc_conv/transpose/ReadVariableOp:value:0 ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: 2
ecc_conv/transpose
ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
ecc_conv/Reshape_2/shape
ecc_conv/Reshape_2Reshapeecc_conv/transpose:y:0!ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: 2
ecc_conv/Reshape_2
ecc_conv/MatMulMatMulecc_conv/Reshape_1:output:0ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv/MatMulz
ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv/Reshape_3/shape/1z
ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape_3/shape/2Ï
ecc_conv/Reshape_3/shapePackecc_conv/unstack:output:0#ecc_conv/Reshape_3/shape/1:output:0#ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape_3/shape§
ecc_conv/Reshape_3Reshapeecc_conv/MatMul:product:0!ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reshape_3
ecc_conv/addAddV2ecc_conv/einsum/Einsum:output:0ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/add§
ecc_conv/BiasAdd/ReadVariableOpReadVariableOp(ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
ecc_conv/BiasAdd/ReadVariableOp 
ecc_conv/BiasAddBiasAddecc_conv/add:z:0'ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/BiasAdd 
ecc_conv/mul_1Mulecc_conv/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/mul_1p
ecc_conv/ReluReluecc_conv/mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reluo
ecc_conv_1/ShapeShapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape
ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2 
ecc_conv_1/strided_slice/stack
 ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice/stack_1
 ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv_1/strided_slice/stack_2¤
ecc_conv_1/strided_sliceStridedSliceecc_conv_1/Shape:output:0'ecc_conv_1/strided_slice/stack:output:0)ecc_conv_1/strided_slice/stack_1:output:0)ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slices
ecc_conv_1/Shape_1Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_1
 ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice_1/stack
"ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/strided_slice_1/stack_1
"ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ecc_conv_1/strided_slice_1/stack_2°
ecc_conv_1/strided_slice_1StridedSliceecc_conv_1/Shape_1:output:0)ecc_conv_1/strided_slice_1/stack:output:0+ecc_conv_1/strided_slice_1/stack_1:output:0+ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slice_1
ecc_conv_1/FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/CastÐ
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02-
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp
!ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!ecc_conv_1/FGN_out/Tensordot/axes
!ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!ecc_conv_1/FGN_out/Tensordot/free
"ecc_conv_1/FGN_out/Tensordot/ShapeShapeecc_conv_1/FGN_out/Cast:y:0*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/Shape
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axis°
%ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/free:output:03ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/GatherV2
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis¶
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:05ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1
"ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/FGN_out/Tensordot/ConstÌ
!ecc_conv_1/FGN_out/Tensordot/ProdProd.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!ecc_conv_1/FGN_out/Tensordot/Prod
$ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$ecc_conv_1/FGN_out/Tensordot/Const_1Ô
#ecc_conv_1/FGN_out/Tensordot/Prod_1Prod0ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#ecc_conv_1/FGN_out/Tensordot/Prod_1
(ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv_1/FGN_out/Tensordot/concat/axis
#ecc_conv_1/FGN_out/Tensordot/concatConcatV2*ecc_conv_1/FGN_out/Tensordot/free:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:01ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv_1/FGN_out/Tensordot/concatØ
"ecc_conv_1/FGN_out/Tensordot/stackPack*ecc_conv_1/FGN_out/Tensordot/Prod:output:0,ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/stackâ
&ecc_conv_1/FGN_out/Tensordot/transpose	Transposeecc_conv_1/FGN_out/Cast:y:0,ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2(
&ecc_conv_1/FGN_out/Tensordot/transposeë
$ecc_conv_1/FGN_out/Tensordot/ReshapeReshape*ecc_conv_1/FGN_out/Tensordot/transpose:y:0+ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$ecc_conv_1/FGN_out/Tensordot/Reshapeë
#ecc_conv_1/FGN_out/Tensordot/MatMulMatMul-ecc_conv_1/FGN_out/Tensordot/Reshape:output:03ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ecc_conv_1/FGN_out/Tensordot/MatMul
$ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$ecc_conv_1/FGN_out/Tensordot/Const_2
*ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/concat_1/axis
%ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_1/FGN_out/Tensordot/Const_2:output:03ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/concat_1á
ecc_conv_1/FGN_out/TensordotReshape-ecc_conv_1/FGN_out/Tensordot/MatMul:product:0.ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/TensordotÆ
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpØ
ecc_conv_1/FGN_out/BiasAddBiasAdd%ecc_conv_1/FGN_out/Tensordot:output:01ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/BiasAdd
ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape/shape/0z
ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape/shape/3
ecc_conv_1/Reshape/shapePack#ecc_conv_1/Reshape/shape/0:output:0!ecc_conv_1/strided_slice:output:0!ecc_conv_1/strided_slice:output:0#ecc_conv_1/Reshape/shape/3:output:0#ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape/shapeÔ
ecc_conv_1/ReshapeReshape#ecc_conv_1/FGN_out/BiasAdd:output:0!ecc_conv_1/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape
 ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv_1/strided_slice_2/stack
"ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2$
"ecc_conv_1/strided_slice_2/stack_1
"ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2$
"ecc_conv_1/strided_slice_2/stack_2Ì
ecc_conv_1/strided_slice_2StridedSliceinputs_1)ecc_conv_1/strided_slice_2/stack:output:0+ecc_conv_1/strided_slice_2/stack_1:output:0+ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv_1/strided_slice_2°
ecc_conv_1/mulMulecc_conv_1/Reshape:output:0#ecc_conv_1/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/mulÌ
ecc_conv_1/einsum/EinsumEinsumecc_conv_1/mul:z:0ecc_conv/Relu:activations:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv_1/einsum/Einsums
ecc_conv_1/Shape_2Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_2
ecc_conv_1/unstackUnpackecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv_1/unstack±
!ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02#
!ecc_conv_1/Shape_3/ReadVariableOpy
ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        2
ecc_conv_1/Shape_3
ecc_conv_1/unstack_1Unpackecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv_1/unstack_1
ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
ecc_conv_1/Reshape_1/shape«
ecc_conv_1/Reshape_1Reshapeecc_conv/Relu:activations:0#ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/Reshape_1µ
#ecc_conv_1/transpose/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02%
#ecc_conv_1/transpose/ReadVariableOp
ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv_1/transpose/perm³
ecc_conv_1/transpose	Transpose+ecc_conv_1/transpose/ReadVariableOp:value:0"ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/transpose
ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
ecc_conv_1/Reshape_2/shape
ecc_conv_1/Reshape_2Reshapeecc_conv_1/transpose:y:0#ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/Reshape_2 
ecc_conv_1/MatMulMatMulecc_conv_1/Reshape_1:output:0ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/MatMul~
ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv_1/Reshape_3/shape/1~
ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape_3/shape/2Ù
ecc_conv_1/Reshape_3/shapePackecc_conv_1/unstack:output:0%ecc_conv_1/Reshape_3/shape/1:output:0%ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape_3/shape¯
ecc_conv_1/Reshape_3Reshapeecc_conv_1/MatMul:product:0#ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/Reshape_3¡
ecc_conv_1/addAddV2!ecc_conv_1/einsum/Einsum:output:0ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/add­
!ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!ecc_conv_1/BiasAdd/ReadVariableOp¨
ecc_conv_1/BiasAddBiasAddecc_conv_1/add:z:0)ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/BiasAdd¦
ecc_conv_1/mul_1Mulecc_conv_1/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/mul_1
%global_max_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ2'
%global_max_pool/Max/reduction_indices©
global_max_pool/MaxMaxecc_conv_1/mul_1:z:0.global_max_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
global_max_pool/Max 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulglobal_max_pool/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityË
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp ^ecc_conv/BiasAdd/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp*^ecc_conv/FGN_out/Tensordot/ReadVariableOp"^ecc_conv/transpose/ReadVariableOp"^ecc_conv_1/BiasAdd/ReadVariableOp*^ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_1/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2B
ecc_conv/BiasAdd/ReadVariableOpecc_conv/BiasAdd/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2V
)ecc_conv/FGN_out/Tensordot/ReadVariableOp)ecc_conv/FGN_out/Tensordot/ReadVariableOp2F
!ecc_conv/transpose/ReadVariableOp!ecc_conv/transpose/ReadVariableOp2F
!ecc_conv_1/BiasAdd/ReadVariableOp!ecc_conv_1/BiasAdd/ReadVariableOp2V
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_1/transpose/ReadVariableOp#ecc_conv_1/transpose/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2
ï

&__inference_dense_layer_call_fn_151877

inputs
unknown:	 
	unknown_0:	
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1505192
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ò
L
0__inference_global_max_pool_layer_call_fn_151857

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_global_max_pool_layer_call_and_return_conditional_losses_1505062
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
ø
ú
+__inference_ecc_conv_1_layer_call_fn_151846
inputs_0
inputs_1
inputs_2	

mask_0
unknown:	
	unknown_0:	
	unknown_1:  
	unknown_2: 
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2mask_0unknown	unknown_0	unknown_1	unknown_2*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_ecc_conv_1_layer_call_and_return_conditional_losses_1504902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namemask/0
ã

$__inference_net_layer_call_fn_150920
inputs_0
inputs_1
inputs_2	
unknown:	 
	unknown_0:	 
	unknown_1: 
	unknown_2: 
	unknown_3:	
	unknown_4:	
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:	
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_net_layer_call_and_return_conditional_losses_1506842
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2
Ö
J
.__inference_graph_masking_layer_call_fn_151653

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_graph_masking_layer_call_and_return_conditional_losses_1506152
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ô
ø
)__inference_ecc_conv_layer_call_fn_151750
inputs_0
inputs_1
inputs_2	

mask_0
unknown:	 
	unknown_0:	 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2mask_0unknown	unknown_0	unknown_1	unknown_2*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_ecc_conv_layer_call_and_return_conditional_losses_1504002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namemask/0
·X
Ý
D__inference_ecc_conv_layer_call_and_return_conditional_losses_151734
inputs_0
inputs_1
inputs_2	

mask_0<
)fgn_out_tensordot_readvariableop_resource:	 6
'fgn_out_biasadd_readvariableop_resource:	 1
shape_3_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢FGN_out/BiasAdd/ReadVariableOp¢ FGN_out/Tensordot/ReadVariableOp¢transpose/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1w
FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Cast¯
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 FGN_out/Tensordot/ReadVariableOpz
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
FGN_out/Tensordot/axes
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
FGN_out/Tensordot/freer
FGN_out/Tensordot/ShapeShapeFGN_out/Cast:y:0*
T0*
_output_shapes
:2
FGN_out/Tensordot/Shape
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/GatherV2/axisù
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!FGN_out/Tensordot/GatherV2_1/axisÿ
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2_1|
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const 
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const_1¨
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod_1
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
FGN_out/Tensordot/concat/axisØ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat¬
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/stack¶
FGN_out/Tensordot/transpose	TransposeFGN_out/Cast:y:0!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Tensordot/transpose¿
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
FGN_out/Tensordot/Reshape¿
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
FGN_out/Tensordot/MatMul
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const_2
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/concat_1/axiså
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat_1µ
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
FGN_out/Tensordot¥
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02 
FGN_out/BiasAdd/ReadVariableOp¬
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
FGN_out/BiasAddm
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3Ò
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape¨
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2	
Reshape
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_2/stack_2
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
strided_slice_2
mulMulReshape:output:0strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
mul
einsum/EinsumEinsummul:z:0inputs_0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
einsum/EinsumJ
Shape_2Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_1/shapew
	Reshape_1Reshapeinputs_0Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_1
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_2/shapes
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_2t
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMulh
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape_3/shape/1h
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_3/shape/2¢
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_3/shape
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
	Reshape_3u
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp|
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2	
BiasAdde
mul_1MulBiasAdd:output:0mask_0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
mul_1U
ReluRelu	mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

IdentityÆ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ
: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_namemask/0

ô
A__inference_dense_layer_call_and_return_conditional_losses_150519

inputs1
matmul_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
X
Ù
D__inference_ecc_conv_layer_call_and_return_conditional_losses_150400

inputs
inputs_1
inputs_2	
mask<
)fgn_out_tensordot_readvariableop_resource:	 6
'fgn_out_biasadd_readvariableop_resource:	 1
shape_3_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢FGN_out/BiasAdd/ReadVariableOp¢ FGN_out/Tensordot/ReadVariableOp¢transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1w
FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Cast¯
 FGN_out/Tensordot/ReadVariableOpReadVariableOp)fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 FGN_out/Tensordot/ReadVariableOpz
FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
FGN_out/Tensordot/axes
FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
FGN_out/Tensordot/freer
FGN_out/Tensordot/ShapeShapeFGN_out/Cast:y:0*
T0*
_output_shapes
:2
FGN_out/Tensordot/Shape
FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/GatherV2/axisù
FGN_out/Tensordot/GatherV2GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/free:output:0(FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2
!FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!FGN_out/Tensordot/GatherV2_1/axisÿ
FGN_out/Tensordot/GatherV2_1GatherV2 FGN_out/Tensordot/Shape:output:0FGN_out/Tensordot/axes:output:0*FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
FGN_out/Tensordot/GatherV2_1|
FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const 
FGN_out/Tensordot/ProdProd#FGN_out/Tensordot/GatherV2:output:0 FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod
FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const_1¨
FGN_out/Tensordot/Prod_1Prod%FGN_out/Tensordot/GatherV2_1:output:0"FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
FGN_out/Tensordot/Prod_1
FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
FGN_out/Tensordot/concat/axisØ
FGN_out/Tensordot/concatConcatV2FGN_out/Tensordot/free:output:0FGN_out/Tensordot/axes:output:0&FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat¬
FGN_out/Tensordot/stackPackFGN_out/Tensordot/Prod:output:0!FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/stack¶
FGN_out/Tensordot/transpose	TransposeFGN_out/Cast:y:0!FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
FGN_out/Tensordot/transpose¿
FGN_out/Tensordot/ReshapeReshapeFGN_out/Tensordot/transpose:y:0 FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
FGN_out/Tensordot/Reshape¿
FGN_out/Tensordot/MatMulMatMul"FGN_out/Tensordot/Reshape:output:0(FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
FGN_out/Tensordot/MatMul
FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
FGN_out/Tensordot/Const_2
FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
FGN_out/Tensordot/concat_1/axiså
FGN_out/Tensordot/concat_1ConcatV2#FGN_out/Tensordot/GatherV2:output:0"FGN_out/Tensordot/Const_2:output:0(FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
FGN_out/Tensordot/concat_1µ
FGN_out/TensordotReshape"FGN_out/Tensordot/MatMul:product:0#FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
FGN_out/Tensordot¥
FGN_out/BiasAdd/ReadVariableOpReadVariableOp'fgn_out_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02 
FGN_out/BiasAdd/ReadVariableOp¬
FGN_out/BiasAddBiasAddFGN_out/Tensordot:output:0&FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
FGN_out/BiasAddm
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3Ò
Reshape/shapePackReshape/shape/0:output:0strided_slice:output:0strided_slice:output:0Reshape/shape/3:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape¨
ReshapeReshapeFGN_out/BiasAdd:output:0Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2	
Reshape
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_2/stack_2
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
strided_slice_2
mulMulReshape:output:0strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
mul
einsum/EinsumEinsummul:z:0inputs*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
einsum/EinsumH
Shape_2Shapeinputs*
T0*
_output_shapes
:2	
Shape_2^
unstackUnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_3`
	unstack_1UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_1/shapeu
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_1
transpose/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_2/shapes
	Reshape_2Reshapetranspose:y:0Reshape_2/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_2t
MatMulMatMulReshape_1:output:0Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMulh
Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape_3/shape/1h
Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_3/shape/2¢
Reshape_3/shapePackunstack:output:0Reshape_3/shape/1:output:0Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_3/shape
	Reshape_3ReshapeMatMul:product:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
	Reshape_3u
addAddV2einsum/Einsum:output:0Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp|
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2	
BiasAddc
mul_1MulBiasAdd:output:0mask*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
mul_1U
ReluRelu	mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

IdentityÆ
NoOpNoOp^BiasAdd/ReadVariableOp^FGN_out/BiasAdd/ReadVariableOp!^FGN_out/Tensordot/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ
: : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2@
FGN_out/BiasAdd/ReadVariableOpFGN_out/BiasAdd/ReadVariableOp2D
 FGN_out/Tensordot/ReadVariableOp FGN_out/Tensordot/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs:QM
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namemask
¶æ
è
?__inference_net_layer_call_and_return_conditional_losses_151117
inputs_0
inputs_1
inputs_2	E
2ecc_conv_fgn_out_tensordot_readvariableop_resource:	 ?
0ecc_conv_fgn_out_biasadd_readvariableop_resource:	 :
(ecc_conv_shape_3_readvariableop_resource: 6
(ecc_conv_biasadd_readvariableop_resource: G
4ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	A
2ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	<
*ecc_conv_1_shape_3_readvariableop_resource:  8
*ecc_conv_1_biasadd_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	 4
%dense_biasadd_readvariableop_resource:	
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢ecc_conv/BiasAdd/ReadVariableOp¢'ecc_conv/FGN_out/BiasAdd/ReadVariableOp¢)ecc_conv/FGN_out/Tensordot/ReadVariableOp¢!ecc_conv/transpose/ReadVariableOp¢!ecc_conv_1/BiasAdd/ReadVariableOp¢)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp¢+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp¢#ecc_conv_1/transpose/ReadVariableOp
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!graph_masking/strided_slice/stack
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice/stack_1
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#graph_masking/strided_slice/stack_2Æ
graph_masking/strided_sliceStridedSliceinputs_0*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
graph_masking/strided_slice
#graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice_1/stack
%graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%graph_masking/strided_slice_1/stack_1
%graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%graph_masking/strided_slice_1/stack_2Î
graph_masking/strided_slice_1StridedSliceinputs_0,graph_masking/strided_slice_1/stack:output:0.graph_masking/strided_slice_1/stack_1:output:0.graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ellipsis_mask*
end_mask2
graph_masking/strided_slice_1t
ecc_conv/ShapeShape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
ecc_conv/strided_slice/stack
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice/stack_1
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slicex
ecc_conv/Shape_1Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_1
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice_1/stack
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/strided_slice_1/stack_1
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv/strided_slice_1/stack_2¤
ecc_conv/strided_slice_1StridedSliceecc_conv/Shape_1:output:0'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice_1
ecc_conv/FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv/FGN_out/CastÊ
)ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp2ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype02+
)ecc_conv/FGN_out/Tensordot/ReadVariableOp
ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
ecc_conv/FGN_out/Tensordot/axes
ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
ecc_conv/FGN_out/Tensordot/free
 ecc_conv/FGN_out/Tensordot/ShapeShapeecc_conv/FGN_out/Cast:y:0*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/Shape
(ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/GatherV2/axis¦
#ecc_conv/FGN_out/Tensordot/GatherV2GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/free:output:01ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/GatherV2
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axis¬
%ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/axes:output:03ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv/FGN_out/Tensordot/GatherV2_1
 ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/FGN_out/Tensordot/ConstÄ
ecc_conv/FGN_out/Tensordot/ProdProd,ecc_conv/FGN_out/Tensordot/GatherV2:output:0)ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
ecc_conv/FGN_out/Tensordot/Prod
"ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_1Ì
!ecc_conv/FGN_out/Tensordot/Prod_1Prod.ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0+ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!ecc_conv/FGN_out/Tensordot/Prod_1
&ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&ecc_conv/FGN_out/Tensordot/concat/axis
!ecc_conv/FGN_out/Tensordot/concatConcatV2(ecc_conv/FGN_out/Tensordot/free:output:0(ecc_conv/FGN_out/Tensordot/axes:output:0/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!ecc_conv/FGN_out/Tensordot/concatÐ
 ecc_conv/FGN_out/Tensordot/stackPack(ecc_conv/FGN_out/Tensordot/Prod:output:0*ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/stackÚ
$ecc_conv/FGN_out/Tensordot/transpose	Transposeecc_conv/FGN_out/Cast:y:0*ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2&
$ecc_conv/FGN_out/Tensordot/transposeã
"ecc_conv/FGN_out/Tensordot/ReshapeReshape(ecc_conv/FGN_out/Tensordot/transpose:y:0)ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"ecc_conv/FGN_out/Tensordot/Reshapeã
!ecc_conv/FGN_out/Tensordot/MatMulMatMul+ecc_conv/FGN_out/Tensordot/Reshape:output:01ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ecc_conv/FGN_out/Tensordot/MatMul
"ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_2
(ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/concat_1/axis
#ecc_conv/FGN_out/Tensordot/concat_1ConcatV2,ecc_conv/FGN_out/Tensordot/GatherV2:output:0+ecc_conv/FGN_out/Tensordot/Const_2:output:01ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/concat_1Ù
ecc_conv/FGN_out/TensordotReshape+ecc_conv/FGN_out/Tensordot/MatMul:product:0,ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/TensordotÀ
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpÐ
ecc_conv/FGN_out/BiasAddBiasAdd#ecc_conv/FGN_out/Tensordot:output:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/BiasAdd
ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape/shape/0v
ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape/shape/3
ecc_conv/Reshape/shapePack!ecc_conv/Reshape/shape/0:output:0ecc_conv/strided_slice:output:0ecc_conv/strided_slice:output:0!ecc_conv/Reshape/shape/3:output:0!ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape/shapeÌ
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_2/stack
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_2/stack_1
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_2/stack_2Â
ecc_conv/strided_slice_2StridedSliceinputs_1'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv/strided_slice_2¨
ecc_conv/mulMulecc_conv/Reshape:output:0!ecc_conv/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv/mulÏ
ecc_conv/einsum/EinsumEinsumecc_conv/mul:z:0$graph_masking/strided_slice:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv/einsum/Einsumx
ecc_conv/Shape_2Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_2y
ecc_conv/unstackUnpackecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv/unstack«
ecc_conv/Shape_3/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02!
ecc_conv/Shape_3/ReadVariableOpu
ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/Shape_3{
ecc_conv/unstack_1Unpackecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv/unstack_1
ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
ecc_conv/Reshape_1/shape®
ecc_conv/Reshape_1Reshape$graph_masking/strided_slice:output:0!ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape_1¯
!ecc_conv/transpose/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02#
!ecc_conv/transpose/ReadVariableOp
ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/transpose/perm«
ecc_conv/transpose	Transpose)ecc_conv/transpose/ReadVariableOp:value:0 ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: 2
ecc_conv/transpose
ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
ecc_conv/Reshape_2/shape
ecc_conv/Reshape_2Reshapeecc_conv/transpose:y:0!ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: 2
ecc_conv/Reshape_2
ecc_conv/MatMulMatMulecc_conv/Reshape_1:output:0ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv/MatMulz
ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv/Reshape_3/shape/1z
ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape_3/shape/2Ï
ecc_conv/Reshape_3/shapePackecc_conv/unstack:output:0#ecc_conv/Reshape_3/shape/1:output:0#ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape_3/shape§
ecc_conv/Reshape_3Reshapeecc_conv/MatMul:product:0!ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reshape_3
ecc_conv/addAddV2ecc_conv/einsum/Einsum:output:0ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/add§
ecc_conv/BiasAdd/ReadVariableOpReadVariableOp(ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
ecc_conv/BiasAdd/ReadVariableOp 
ecc_conv/BiasAddBiasAddecc_conv/add:z:0'ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/BiasAdd 
ecc_conv/mul_1Mulecc_conv/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/mul_1p
ecc_conv/ReluReluecc_conv/mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reluo
ecc_conv_1/ShapeShapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape
ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2 
ecc_conv_1/strided_slice/stack
 ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice/stack_1
 ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv_1/strided_slice/stack_2¤
ecc_conv_1/strided_sliceStridedSliceecc_conv_1/Shape:output:0'ecc_conv_1/strided_slice/stack:output:0)ecc_conv_1/strided_slice/stack_1:output:0)ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slices
ecc_conv_1/Shape_1Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_1
 ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice_1/stack
"ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/strided_slice_1/stack_1
"ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ecc_conv_1/strided_slice_1/stack_2°
ecc_conv_1/strided_slice_1StridedSliceecc_conv_1/Shape_1:output:0)ecc_conv_1/strided_slice_1/stack:output:0+ecc_conv_1/strided_slice_1/stack_1:output:0+ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slice_1
ecc_conv_1/FGN_out/CastCastinputs_2*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/CastÐ
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02-
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp
!ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!ecc_conv_1/FGN_out/Tensordot/axes
!ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!ecc_conv_1/FGN_out/Tensordot/free
"ecc_conv_1/FGN_out/Tensordot/ShapeShapeecc_conv_1/FGN_out/Cast:y:0*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/Shape
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axis°
%ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/free:output:03ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/GatherV2
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis¶
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:05ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1
"ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/FGN_out/Tensordot/ConstÌ
!ecc_conv_1/FGN_out/Tensordot/ProdProd.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!ecc_conv_1/FGN_out/Tensordot/Prod
$ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$ecc_conv_1/FGN_out/Tensordot/Const_1Ô
#ecc_conv_1/FGN_out/Tensordot/Prod_1Prod0ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#ecc_conv_1/FGN_out/Tensordot/Prod_1
(ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv_1/FGN_out/Tensordot/concat/axis
#ecc_conv_1/FGN_out/Tensordot/concatConcatV2*ecc_conv_1/FGN_out/Tensordot/free:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:01ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv_1/FGN_out/Tensordot/concatØ
"ecc_conv_1/FGN_out/Tensordot/stackPack*ecc_conv_1/FGN_out/Tensordot/Prod:output:0,ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/stackâ
&ecc_conv_1/FGN_out/Tensordot/transpose	Transposeecc_conv_1/FGN_out/Cast:y:0,ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2(
&ecc_conv_1/FGN_out/Tensordot/transposeë
$ecc_conv_1/FGN_out/Tensordot/ReshapeReshape*ecc_conv_1/FGN_out/Tensordot/transpose:y:0+ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$ecc_conv_1/FGN_out/Tensordot/Reshapeë
#ecc_conv_1/FGN_out/Tensordot/MatMulMatMul-ecc_conv_1/FGN_out/Tensordot/Reshape:output:03ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ecc_conv_1/FGN_out/Tensordot/MatMul
$ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$ecc_conv_1/FGN_out/Tensordot/Const_2
*ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/concat_1/axis
%ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_1/FGN_out/Tensordot/Const_2:output:03ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/concat_1á
ecc_conv_1/FGN_out/TensordotReshape-ecc_conv_1/FGN_out/Tensordot/MatMul:product:0.ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/TensordotÆ
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpØ
ecc_conv_1/FGN_out/BiasAddBiasAdd%ecc_conv_1/FGN_out/Tensordot:output:01ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/BiasAdd
ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape/shape/0z
ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape/shape/3
ecc_conv_1/Reshape/shapePack#ecc_conv_1/Reshape/shape/0:output:0!ecc_conv_1/strided_slice:output:0!ecc_conv_1/strided_slice:output:0#ecc_conv_1/Reshape/shape/3:output:0#ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape/shapeÔ
ecc_conv_1/ReshapeReshape#ecc_conv_1/FGN_out/BiasAdd:output:0!ecc_conv_1/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape
 ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv_1/strided_slice_2/stack
"ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2$
"ecc_conv_1/strided_slice_2/stack_1
"ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2$
"ecc_conv_1/strided_slice_2/stack_2Ì
ecc_conv_1/strided_slice_2StridedSliceinputs_1)ecc_conv_1/strided_slice_2/stack:output:0+ecc_conv_1/strided_slice_2/stack_1:output:0+ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv_1/strided_slice_2°
ecc_conv_1/mulMulecc_conv_1/Reshape:output:0#ecc_conv_1/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/mulÌ
ecc_conv_1/einsum/EinsumEinsumecc_conv_1/mul:z:0ecc_conv/Relu:activations:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv_1/einsum/Einsums
ecc_conv_1/Shape_2Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_2
ecc_conv_1/unstackUnpackecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv_1/unstack±
!ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02#
!ecc_conv_1/Shape_3/ReadVariableOpy
ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        2
ecc_conv_1/Shape_3
ecc_conv_1/unstack_1Unpackecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv_1/unstack_1
ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
ecc_conv_1/Reshape_1/shape«
ecc_conv_1/Reshape_1Reshapeecc_conv/Relu:activations:0#ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/Reshape_1µ
#ecc_conv_1/transpose/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02%
#ecc_conv_1/transpose/ReadVariableOp
ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv_1/transpose/perm³
ecc_conv_1/transpose	Transpose+ecc_conv_1/transpose/ReadVariableOp:value:0"ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/transpose
ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
ecc_conv_1/Reshape_2/shape
ecc_conv_1/Reshape_2Reshapeecc_conv_1/transpose:y:0#ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/Reshape_2 
ecc_conv_1/MatMulMatMulecc_conv_1/Reshape_1:output:0ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/MatMul~
ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv_1/Reshape_3/shape/1~
ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape_3/shape/2Ù
ecc_conv_1/Reshape_3/shapePackecc_conv_1/unstack:output:0%ecc_conv_1/Reshape_3/shape/1:output:0%ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape_3/shape¯
ecc_conv_1/Reshape_3Reshapeecc_conv_1/MatMul:product:0#ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/Reshape_3¡
ecc_conv_1/addAddV2!ecc_conv_1/einsum/Einsum:output:0ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/add­
!ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!ecc_conv_1/BiasAdd/ReadVariableOp¨
ecc_conv_1/BiasAddBiasAddecc_conv_1/add:z:0)ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/BiasAdd¦
ecc_conv_1/mul_1Mulecc_conv_1/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/mul_1
%global_max_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ2'
%global_max_pool/Max/reduction_indices©
global_max_pool/MaxMaxecc_conv_1/mul_1:z:0.global_max_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
global_max_pool/Max 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulglobal_max_pool/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityË
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp ^ecc_conv/BiasAdd/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp*^ecc_conv/FGN_out/Tensordot/ReadVariableOp"^ecc_conv/transpose/ReadVariableOp"^ecc_conv_1/BiasAdd/ReadVariableOp*^ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_1/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2B
ecc_conv/BiasAdd/ReadVariableOpecc_conv/BiasAdd/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2V
)ecc_conv/FGN_out/Tensordot/ReadVariableOp)ecc_conv/FGN_out/Tensordot/ReadVariableOp2F
!ecc_conv/transpose/ReadVariableOp!ecc_conv/transpose/ReadVariableOp2F
!ecc_conv_1/BiasAdd/ReadVariableOp!ecc_conv_1/BiasAdd/ReadVariableOp2V
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_1/transpose/ReadVariableOp#ecc_conv_1/transpose/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/1:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


"
_user_specified_name
inputs/2
$
´
?__inference_net_layer_call_and_return_conditional_losses_150684

inputs
inputs_1
inputs_2	"
ecc_conv_150659:	 
ecc_conv_150661:	 !
ecc_conv_150663: 
ecc_conv_150665: $
ecc_conv_1_150668:	 
ecc_conv_1_150670:	#
ecc_conv_1_150672:  
ecc_conv_1_150674: 
dense_150678:	 
dense_150680:	
identity¢dense/StatefulPartitionedCall¢ ecc_conv/StatefulPartitionedCall¢"ecc_conv_1/StatefulPartitionedCallç
graph_masking/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_graph_masking_layer_call_and_return_conditional_losses_1506152
graph_masking/PartitionedCall
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2#
!graph_masking/strided_slice/stack
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#graph_masking/strided_slice/stack_1
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#graph_masking/strided_slice/stack_2Â
graph_masking/strided_sliceStridedSliceinputs*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ellipsis_mask*
end_mask2
graph_masking/strided_slice
 ecc_conv/StatefulPartitionedCallStatefulPartitionedCall&graph_masking/PartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_150659ecc_conv_150661ecc_conv_150663ecc_conv_150665*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_ecc_conv_layer_call_and_return_conditional_losses_1504002"
 ecc_conv/StatefulPartitionedCall¬
"ecc_conv_1/StatefulPartitionedCallStatefulPartitionedCall)ecc_conv/StatefulPartitionedCall:output:0inputs_1inputs_2$graph_masking/strided_slice:output:0ecc_conv_1_150668ecc_conv_1_150670ecc_conv_1_150672ecc_conv_1_150674*
Tin

2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_ecc_conv_1_layer_call_and_return_conditional_losses_1504902$
"ecc_conv_1/StatefulPartitionedCall
global_max_pool/PartitionedCallPartitionedCall+ecc_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_global_max_pool_layer_call_and_return_conditional_losses_1505062!
global_max_pool/PartitionedCall¨
dense/StatefulPartitionedCallStatefulPartitionedCall(global_max_pool/PartitionedCall:output:0dense_150678dense_150680*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1505192
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¶
NoOpNoOp^dense/StatefulPartitionedCall!^ecc_conv/StatefulPartitionedCall#^ecc_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 ecc_conv/StatefulPartitionedCall ecc_conv/StatefulPartitionedCall2H
"ecc_conv_1/StatefulPartitionedCall"ecc_conv_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
¡
e
I__inference_graph_masking_layer_call_and_return_conditional_losses_151635

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Æö
	
!__inference__wrapped_model_150294
input_1
input_2
input_3	I
6net_ecc_conv_fgn_out_tensordot_readvariableop_resource:	 C
4net_ecc_conv_fgn_out_biasadd_readvariableop_resource:	 >
,net_ecc_conv_shape_3_readvariableop_resource: :
,net_ecc_conv_biasadd_readvariableop_resource: K
8net_ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	E
6net_ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	@
.net_ecc_conv_1_shape_3_readvariableop_resource:  <
.net_ecc_conv_1_biasadd_readvariableop_resource: ;
(net_dense_matmul_readvariableop_resource:	 8
)net_dense_biasadd_readvariableop_resource:	
identity¢ net/dense/BiasAdd/ReadVariableOp¢net/dense/MatMul/ReadVariableOp¢#net/ecc_conv/BiasAdd/ReadVariableOp¢+net/ecc_conv/FGN_out/BiasAdd/ReadVariableOp¢-net/ecc_conv/FGN_out/Tensordot/ReadVariableOp¢%net/ecc_conv/transpose/ReadVariableOp¢%net/ecc_conv_1/BiasAdd/ReadVariableOp¢-net/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp¢/net/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp¢'net/ecc_conv_1/transpose/ReadVariableOp
%net/graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%net/graph_masking/strided_slice/stack£
'net/graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2)
'net/graph_masking/strided_slice/stack_1£
'net/graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'net/graph_masking/strided_slice/stack_2Ù
net/graph_masking/strided_sliceStridedSliceinput_1.net/graph_masking/strided_slice/stack:output:00net/graph_masking/strided_slice/stack_1:output:00net/graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2!
net/graph_masking/strided_slice£
'net/graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2)
'net/graph_masking/strided_slice_1/stack§
)net/graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)net/graph_masking/strided_slice_1/stack_1§
)net/graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)net/graph_masking/strided_slice_1/stack_2á
!net/graph_masking/strided_slice_1StridedSliceinput_10net/graph_masking/strided_slice_1/stack:output:02net/graph_masking/strided_slice_1/stack_1:output:02net/graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ellipsis_mask*
end_mask2#
!net/graph_masking/strided_slice_1
net/ecc_conv/ShapeShape(net/graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
net/ecc_conv/Shape
 net/ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2"
 net/ecc_conv/strided_slice/stack
"net/ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2$
"net/ecc_conv/strided_slice/stack_1
"net/ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"net/ecc_conv/strided_slice/stack_2°
net/ecc_conv/strided_sliceStridedSlicenet/ecc_conv/Shape:output:0)net/ecc_conv/strided_slice/stack:output:0+net/ecc_conv/strided_slice/stack_1:output:0+net/ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
net/ecc_conv/strided_slice
net/ecc_conv/Shape_1Shape(net/graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
net/ecc_conv/Shape_1
"net/ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2$
"net/ecc_conv/strided_slice_1/stack
$net/ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$net/ecc_conv/strided_slice_1/stack_1
$net/ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$net/ecc_conv/strided_slice_1/stack_2¼
net/ecc_conv/strided_slice_1StridedSlicenet/ecc_conv/Shape_1:output:0+net/ecc_conv/strided_slice_1/stack:output:0-net/ecc_conv/strided_slice_1/stack_1:output:0-net/ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
net/ecc_conv/strided_slice_1
net/ecc_conv/FGN_out/CastCastinput_3*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
net/ecc_conv/FGN_out/CastÖ
-net/ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp6net_ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-net/ecc_conv/FGN_out/Tensordot/ReadVariableOp
#net/ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#net/ecc_conv/FGN_out/Tensordot/axes
#net/ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#net/ecc_conv/FGN_out/Tensordot/free
$net/ecc_conv/FGN_out/Tensordot/ShapeShapenet/ecc_conv/FGN_out/Cast:y:0*
T0*
_output_shapes
:2&
$net/ecc_conv/FGN_out/Tensordot/Shape
,net/ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,net/ecc_conv/FGN_out/Tensordot/GatherV2/axisº
'net/ecc_conv/FGN_out/Tensordot/GatherV2GatherV2-net/ecc_conv/FGN_out/Tensordot/Shape:output:0,net/ecc_conv/FGN_out/Tensordot/free:output:05net/ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'net/ecc_conv/FGN_out/Tensordot/GatherV2¢
.net/ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.net/ecc_conv/FGN_out/Tensordot/GatherV2_1/axisÀ
)net/ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2-net/ecc_conv/FGN_out/Tensordot/Shape:output:0,net/ecc_conv/FGN_out/Tensordot/axes:output:07net/ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)net/ecc_conv/FGN_out/Tensordot/GatherV2_1
$net/ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$net/ecc_conv/FGN_out/Tensordot/ConstÔ
#net/ecc_conv/FGN_out/Tensordot/ProdProd0net/ecc_conv/FGN_out/Tensordot/GatherV2:output:0-net/ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#net/ecc_conv/FGN_out/Tensordot/Prod
&net/ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&net/ecc_conv/FGN_out/Tensordot/Const_1Ü
%net/ecc_conv/FGN_out/Tensordot/Prod_1Prod2net/ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0/net/ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%net/ecc_conv/FGN_out/Tensordot/Prod_1
*net/ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*net/ecc_conv/FGN_out/Tensordot/concat/axis
%net/ecc_conv/FGN_out/Tensordot/concatConcatV2,net/ecc_conv/FGN_out/Tensordot/free:output:0,net/ecc_conv/FGN_out/Tensordot/axes:output:03net/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%net/ecc_conv/FGN_out/Tensordot/concatà
$net/ecc_conv/FGN_out/Tensordot/stackPack,net/ecc_conv/FGN_out/Tensordot/Prod:output:0.net/ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$net/ecc_conv/FGN_out/Tensordot/stackê
(net/ecc_conv/FGN_out/Tensordot/transpose	Transposenet/ecc_conv/FGN_out/Cast:y:0.net/ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2*
(net/ecc_conv/FGN_out/Tensordot/transposeó
&net/ecc_conv/FGN_out/Tensordot/ReshapeReshape,net/ecc_conv/FGN_out/Tensordot/transpose:y:0-net/ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&net/ecc_conv/FGN_out/Tensordot/Reshapeó
%net/ecc_conv/FGN_out/Tensordot/MatMulMatMul/net/ecc_conv/FGN_out/Tensordot/Reshape:output:05net/ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%net/ecc_conv/FGN_out/Tensordot/MatMul
&net/ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&net/ecc_conv/FGN_out/Tensordot/Const_2
,net/ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,net/ecc_conv/FGN_out/Tensordot/concat_1/axis¦
'net/ecc_conv/FGN_out/Tensordot/concat_1ConcatV20net/ecc_conv/FGN_out/Tensordot/GatherV2:output:0/net/ecc_conv/FGN_out/Tensordot/Const_2:output:05net/ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'net/ecc_conv/FGN_out/Tensordot/concat_1é
net/ecc_conv/FGN_out/TensordotReshape/net/ecc_conv/FGN_out/Tensordot/MatMul:product:00net/ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2 
net/ecc_conv/FGN_out/TensordotÌ
+net/ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp4net_ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02-
+net/ecc_conv/FGN_out/BiasAdd/ReadVariableOpà
net/ecc_conv/FGN_out/BiasAddBiasAdd'net/ecc_conv/FGN_out/Tensordot:output:03net/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
net/ecc_conv/FGN_out/BiasAdd
net/ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
net/ecc_conv/Reshape/shape/0~
net/ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
net/ecc_conv/Reshape/shape/3­
net/ecc_conv/Reshape/shapePack%net/ecc_conv/Reshape/shape/0:output:0#net/ecc_conv/strided_slice:output:0#net/ecc_conv/strided_slice:output:0%net/ecc_conv/Reshape/shape/3:output:0%net/ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
net/ecc_conv/Reshape/shapeÜ
net/ecc_conv/ReshapeReshape%net/ecc_conv/FGN_out/BiasAdd:output:0#net/ecc_conv/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
net/ecc_conv/Reshape
"net/ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2$
"net/ecc_conv/strided_slice_2/stack¡
$net/ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2&
$net/ecc_conv/strided_slice_2/stack_1¡
$net/ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2&
$net/ecc_conv/strided_slice_2/stack_2Õ
net/ecc_conv/strided_slice_2StridedSliceinput_2+net/ecc_conv/strided_slice_2/stack:output:0-net/ecc_conv/strided_slice_2/stack_1:output:0-net/ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
net/ecc_conv/strided_slice_2¸
net/ecc_conv/mulMulnet/ecc_conv/Reshape:output:0%net/ecc_conv/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
net/ecc_conv/mulß
net/ecc_conv/einsum/EinsumEinsumnet/ecc_conv/mul:z:0(net/graph_masking/strided_slice:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
net/ecc_conv/einsum/Einsum
net/ecc_conv/Shape_2Shape(net/graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
net/ecc_conv/Shape_2
net/ecc_conv/unstackUnpacknet/ecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
net/ecc_conv/unstack·
#net/ecc_conv/Shape_3/ReadVariableOpReadVariableOp,net_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02%
#net/ecc_conv/Shape_3/ReadVariableOp}
net/ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       2
net/ecc_conv/Shape_3
net/ecc_conv/unstack_1Unpacknet/ecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
net/ecc_conv/unstack_1
net/ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
net/ecc_conv/Reshape_1/shape¾
net/ecc_conv/Reshape_1Reshape(net/graph_masking/strided_slice:output:0%net/ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
net/ecc_conv/Reshape_1»
%net/ecc_conv/transpose/ReadVariableOpReadVariableOp,net_ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02'
%net/ecc_conv/transpose/ReadVariableOp
net/ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
net/ecc_conv/transpose/perm»
net/ecc_conv/transpose	Transpose-net/ecc_conv/transpose/ReadVariableOp:value:0$net/ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: 2
net/ecc_conv/transpose
net/ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
net/ecc_conv/Reshape_2/shape§
net/ecc_conv/Reshape_2Reshapenet/ecc_conv/transpose:y:0%net/ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: 2
net/ecc_conv/Reshape_2¨
net/ecc_conv/MatMulMatMulnet/ecc_conv/Reshape_1:output:0net/ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
net/ecc_conv/MatMul
net/ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2 
net/ecc_conv/Reshape_3/shape/1
net/ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2 
net/ecc_conv/Reshape_3/shape/2ã
net/ecc_conv/Reshape_3/shapePacknet/ecc_conv/unstack:output:0'net/ecc_conv/Reshape_3/shape/1:output:0'net/ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
net/ecc_conv/Reshape_3/shape·
net/ecc_conv/Reshape_3Reshapenet/ecc_conv/MatMul:product:0%net/ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv/Reshape_3©
net/ecc_conv/addAddV2#net/ecc_conv/einsum/Einsum:output:0net/ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv/add³
#net/ecc_conv/BiasAdd/ReadVariableOpReadVariableOp,net_ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#net/ecc_conv/BiasAdd/ReadVariableOp°
net/ecc_conv/BiasAddBiasAddnet/ecc_conv/add:z:0+net/ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv/BiasAdd°
net/ecc_conv/mul_1Mulnet/ecc_conv/BiasAdd:output:0*net/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv/mul_1|
net/ecc_conv/ReluRelunet/ecc_conv/mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv/Relu{
net/ecc_conv_1/ShapeShapenet/ecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
net/ecc_conv_1/Shape
"net/ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2$
"net/ecc_conv_1/strided_slice/stack
$net/ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$net/ecc_conv_1/strided_slice/stack_1
$net/ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$net/ecc_conv_1/strided_slice/stack_2¼
net/ecc_conv_1/strided_sliceStridedSlicenet/ecc_conv_1/Shape:output:0+net/ecc_conv_1/strided_slice/stack:output:0-net/ecc_conv_1/strided_slice/stack_1:output:0-net/ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
net/ecc_conv_1/strided_slice
net/ecc_conv_1/Shape_1Shapenet/ecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
net/ecc_conv_1/Shape_1
$net/ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$net/ecc_conv_1/strided_slice_1/stack
&net/ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&net/ecc_conv_1/strided_slice_1/stack_1
&net/ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&net/ecc_conv_1/strided_slice_1/stack_2È
net/ecc_conv_1/strided_slice_1StridedSlicenet/ecc_conv_1/Shape_1:output:0-net/ecc_conv_1/strided_slice_1/stack:output:0/net/ecc_conv_1/strided_slice_1/stack_1:output:0/net/ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
net/ecc_conv_1/strided_slice_1
net/ecc_conv_1/FGN_out/CastCastinput_3*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
net/ecc_conv_1/FGN_out/CastÜ
/net/ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOp8net_ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype021
/net/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp
%net/ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%net/ecc_conv_1/FGN_out/Tensordot/axes£
%net/ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%net/ecc_conv_1/FGN_out/Tensordot/free
&net/ecc_conv_1/FGN_out/Tensordot/ShapeShapenet/ecc_conv_1/FGN_out/Cast:y:0*
T0*
_output_shapes
:2(
&net/ecc_conv_1/FGN_out/Tensordot/Shape¢
.net/ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.net/ecc_conv_1/FGN_out/Tensordot/GatherV2/axisÄ
)net/ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2/net/ecc_conv_1/FGN_out/Tensordot/Shape:output:0.net/ecc_conv_1/FGN_out/Tensordot/free:output:07net/ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)net/ecc_conv_1/FGN_out/Tensordot/GatherV2¦
0net/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0net/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisÊ
+net/ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2/net/ecc_conv_1/FGN_out/Tensordot/Shape:output:0.net/ecc_conv_1/FGN_out/Tensordot/axes:output:09net/ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+net/ecc_conv_1/FGN_out/Tensordot/GatherV2_1
&net/ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&net/ecc_conv_1/FGN_out/Tensordot/ConstÜ
%net/ecc_conv_1/FGN_out/Tensordot/ProdProd2net/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0/net/ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%net/ecc_conv_1/FGN_out/Tensordot/Prod
(net/ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(net/ecc_conv_1/FGN_out/Tensordot/Const_1ä
'net/ecc_conv_1/FGN_out/Tensordot/Prod_1Prod4net/ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:01net/ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'net/ecc_conv_1/FGN_out/Tensordot/Prod_1
,net/ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,net/ecc_conv_1/FGN_out/Tensordot/concat/axis£
'net/ecc_conv_1/FGN_out/Tensordot/concatConcatV2.net/ecc_conv_1/FGN_out/Tensordot/free:output:0.net/ecc_conv_1/FGN_out/Tensordot/axes:output:05net/ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'net/ecc_conv_1/FGN_out/Tensordot/concatè
&net/ecc_conv_1/FGN_out/Tensordot/stackPack.net/ecc_conv_1/FGN_out/Tensordot/Prod:output:00net/ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&net/ecc_conv_1/FGN_out/Tensordot/stackò
*net/ecc_conv_1/FGN_out/Tensordot/transpose	Transposenet/ecc_conv_1/FGN_out/Cast:y:00net/ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2,
*net/ecc_conv_1/FGN_out/Tensordot/transposeû
(net/ecc_conv_1/FGN_out/Tensordot/ReshapeReshape.net/ecc_conv_1/FGN_out/Tensordot/transpose:y:0/net/ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(net/ecc_conv_1/FGN_out/Tensordot/Reshapeû
'net/ecc_conv_1/FGN_out/Tensordot/MatMulMatMul1net/ecc_conv_1/FGN_out/Tensordot/Reshape:output:07net/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'net/ecc_conv_1/FGN_out/Tensordot/MatMul
(net/ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(net/ecc_conv_1/FGN_out/Tensordot/Const_2¢
.net/ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.net/ecc_conv_1/FGN_out/Tensordot/concat_1/axis°
)net/ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV22net/ecc_conv_1/FGN_out/Tensordot/GatherV2:output:01net/ecc_conv_1/FGN_out/Tensordot/Const_2:output:07net/ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)net/ecc_conv_1/FGN_out/Tensordot/concat_1ñ
 net/ecc_conv_1/FGN_out/TensordotReshape1net/ecc_conv_1/FGN_out/Tensordot/MatMul:product:02net/ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2"
 net/ecc_conv_1/FGN_out/TensordotÒ
-net/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp6net_ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-net/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpè
net/ecc_conv_1/FGN_out/BiasAddBiasAdd)net/ecc_conv_1/FGN_out/Tensordot:output:05net/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2 
net/ecc_conv_1/FGN_out/BiasAdd
net/ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
net/ecc_conv_1/Reshape/shape/0
net/ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2 
net/ecc_conv_1/Reshape/shape/3»
net/ecc_conv_1/Reshape/shapePack'net/ecc_conv_1/Reshape/shape/0:output:0%net/ecc_conv_1/strided_slice:output:0%net/ecc_conv_1/strided_slice:output:0'net/ecc_conv_1/Reshape/shape/3:output:0'net/ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
net/ecc_conv_1/Reshape/shapeä
net/ecc_conv_1/ReshapeReshape'net/ecc_conv_1/FGN_out/BiasAdd:output:0%net/ecc_conv_1/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
net/ecc_conv_1/Reshape¡
$net/ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2&
$net/ecc_conv_1/strided_slice_2/stack¥
&net/ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2(
&net/ecc_conv_1/strided_slice_2/stack_1¥
&net/ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2(
&net/ecc_conv_1/strided_slice_2/stack_2ß
net/ecc_conv_1/strided_slice_2StridedSliceinput_2-net/ecc_conv_1/strided_slice_2/stack:output:0/net/ecc_conv_1/strided_slice_2/stack_1:output:0/net/ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2 
net/ecc_conv_1/strided_slice_2À
net/ecc_conv_1/mulMulnet/ecc_conv_1/Reshape:output:0'net/ecc_conv_1/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
net/ecc_conv_1/mulÜ
net/ecc_conv_1/einsum/EinsumEinsumnet/ecc_conv_1/mul:z:0net/ecc_conv/Relu:activations:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
net/ecc_conv_1/einsum/Einsum
net/ecc_conv_1/Shape_2Shapenet/ecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
net/ecc_conv_1/Shape_2
net/ecc_conv_1/unstackUnpacknet/ecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
net/ecc_conv_1/unstack½
%net/ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp.net_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02'
%net/ecc_conv_1/Shape_3/ReadVariableOp
net/ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        2
net/ecc_conv_1/Shape_3
net/ecc_conv_1/unstack_1Unpacknet/ecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
net/ecc_conv_1/unstack_1
net/ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2 
net/ecc_conv_1/Reshape_1/shape»
net/ecc_conv_1/Reshape_1Reshapenet/ecc_conv/Relu:activations:0'net/ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
net/ecc_conv_1/Reshape_1Á
'net/ecc_conv_1/transpose/ReadVariableOpReadVariableOp.net_ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02)
'net/ecc_conv_1/transpose/ReadVariableOp
net/ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
net/ecc_conv_1/transpose/permÃ
net/ecc_conv_1/transpose	Transpose/net/ecc_conv_1/transpose/ReadVariableOp:value:0&net/ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  2
net/ecc_conv_1/transpose
net/ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2 
net/ecc_conv_1/Reshape_2/shape¯
net/ecc_conv_1/Reshape_2Reshapenet/ecc_conv_1/transpose:y:0'net/ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  2
net/ecc_conv_1/Reshape_2°
net/ecc_conv_1/MatMulMatMul!net/ecc_conv_1/Reshape_1:output:0!net/ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
net/ecc_conv_1/MatMul
 net/ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2"
 net/ecc_conv_1/Reshape_3/shape/1
 net/ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2"
 net/ecc_conv_1/Reshape_3/shape/2í
net/ecc_conv_1/Reshape_3/shapePacknet/ecc_conv_1/unstack:output:0)net/ecc_conv_1/Reshape_3/shape/1:output:0)net/ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2 
net/ecc_conv_1/Reshape_3/shape¿
net/ecc_conv_1/Reshape_3Reshapenet/ecc_conv_1/MatMul:product:0'net/ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv_1/Reshape_3±
net/ecc_conv_1/addAddV2%net/ecc_conv_1/einsum/Einsum:output:0!net/ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv_1/add¹
%net/ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp.net_ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%net/ecc_conv_1/BiasAdd/ReadVariableOp¸
net/ecc_conv_1/BiasAddBiasAddnet/ecc_conv_1/add:z:0-net/ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv_1/BiasAdd¶
net/ecc_conv_1/mul_1Mulnet/ecc_conv_1/BiasAdd:output:0*net/graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
net/ecc_conv_1/mul_1¡
)net/global_max_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ2+
)net/global_max_pool/Max/reduction_indices¹
net/global_max_pool/MaxMaxnet/ecc_conv_1/mul_1:z:02net/global_max_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
net/global_max_pool/Max¬
net/dense/MatMul/ReadVariableOpReadVariableOp(net_dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02!
net/dense/MatMul/ReadVariableOp¬
net/dense/MatMulMatMul net/global_max_pool/Max:output:0'net/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
net/dense/MatMul«
 net/dense/BiasAdd/ReadVariableOpReadVariableOp)net_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 net/dense/BiasAdd/ReadVariableOpª
net/dense/BiasAddBiasAddnet/dense/MatMul:product:0(net/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
net/dense/BiasAdd
net/dense/SoftmaxSoftmaxnet/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
net/dense/Softmaxw
IdentityIdentitynet/dense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityó
NoOpNoOp!^net/dense/BiasAdd/ReadVariableOp ^net/dense/MatMul/ReadVariableOp$^net/ecc_conv/BiasAdd/ReadVariableOp,^net/ecc_conv/FGN_out/BiasAdd/ReadVariableOp.^net/ecc_conv/FGN_out/Tensordot/ReadVariableOp&^net/ecc_conv/transpose/ReadVariableOp&^net/ecc_conv_1/BiasAdd/ReadVariableOp.^net/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp0^net/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp(^net/ecc_conv_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 2D
 net/dense/BiasAdd/ReadVariableOp net/dense/BiasAdd/ReadVariableOp2B
net/dense/MatMul/ReadVariableOpnet/dense/MatMul/ReadVariableOp2J
#net/ecc_conv/BiasAdd/ReadVariableOp#net/ecc_conv/BiasAdd/ReadVariableOp2Z
+net/ecc_conv/FGN_out/BiasAdd/ReadVariableOp+net/ecc_conv/FGN_out/BiasAdd/ReadVariableOp2^
-net/ecc_conv/FGN_out/Tensordot/ReadVariableOp-net/ecc_conv/FGN_out/Tensordot/ReadVariableOp2N
%net/ecc_conv/transpose/ReadVariableOp%net/ecc_conv/transpose/ReadVariableOp2N
%net/ecc_conv_1/BiasAdd/ReadVariableOp%net/ecc_conv_1/BiasAdd/ReadVariableOp2^
-net/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp-net/ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2b
/net/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp/net/ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2R
'net/ecc_conv_1/transpose/ReadVariableOp'net/ecc_conv_1/transpose/ReadVariableOp:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_2:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_3

ô
A__inference_dense_layer_call_and_return_conditional_losses_151868

inputs1
matmul_readvariableop_resource:	 .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ý
g
K__inference_global_max_pool_layer_call_and_return_conditional_losses_151852

inputs
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ2
Max/reduction_indicesk
MaxMaxinputsMax/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Max`
IdentityIdentityMax:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
ªæ
å
?__inference_net_layer_call_and_return_conditional_losses_151457
input_1
input_2
input_3	E
2ecc_conv_fgn_out_tensordot_readvariableop_resource:	 ?
0ecc_conv_fgn_out_biasadd_readvariableop_resource:	 :
(ecc_conv_shape_3_readvariableop_resource: 6
(ecc_conv_biasadd_readvariableop_resource: G
4ecc_conv_1_fgn_out_tensordot_readvariableop_resource:	A
2ecc_conv_1_fgn_out_biasadd_readvariableop_resource:	<
*ecc_conv_1_shape_3_readvariableop_resource:  8
*ecc_conv_1_biasadd_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:	 4
%dense_biasadd_readvariableop_resource:	
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢ecc_conv/BiasAdd/ReadVariableOp¢'ecc_conv/FGN_out/BiasAdd/ReadVariableOp¢)ecc_conv/FGN_out/Tensordot/ReadVariableOp¢!ecc_conv/transpose/ReadVariableOp¢!ecc_conv_1/BiasAdd/ReadVariableOp¢)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp¢+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp¢#ecc_conv_1/transpose/ReadVariableOp
!graph_masking/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!graph_masking/strided_slice/stack
#graph_masking/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice/stack_1
#graph_masking/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#graph_masking/strided_slice/stack_2Å
graph_masking/strided_sliceStridedSliceinput_1*graph_masking/strided_slice/stack:output:0,graph_masking/strided_slice/stack_1:output:0,graph_masking/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
graph_masking/strided_slice
#graph_masking/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2%
#graph_masking/strided_slice_1/stack
%graph_masking/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%graph_masking/strided_slice_1/stack_1
%graph_masking/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%graph_masking/strided_slice_1/stack_2Í
graph_masking/strided_slice_1StridedSliceinput_1,graph_masking/strided_slice_1/stack:output:0.graph_masking/strided_slice_1/stack_1:output:0.graph_masking/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ellipsis_mask*
end_mask2
graph_masking/strided_slice_1t
ecc_conv/ShapeShape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape
ecc_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
ecc_conv/strided_slice/stack
ecc_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice/stack_1
ecc_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
ecc_conv/strided_slice/stack_2
ecc_conv/strided_sliceStridedSliceecc_conv/Shape:output:0%ecc_conv/strided_slice/stack:output:0'ecc_conv/strided_slice/stack_1:output:0'ecc_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slicex
ecc_conv/Shape_1Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_1
ecc_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
ecc_conv/strided_slice_1/stack
 ecc_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/strided_slice_1/stack_1
 ecc_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv/strided_slice_1/stack_2¤
ecc_conv/strided_slice_1StridedSliceecc_conv/Shape_1:output:0'ecc_conv/strided_slice_1/stack:output:0)ecc_conv/strided_slice_1/stack_1:output:0)ecc_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv/strided_slice_1
ecc_conv/FGN_out/CastCastinput_3*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv/FGN_out/CastÊ
)ecc_conv/FGN_out/Tensordot/ReadVariableOpReadVariableOp2ecc_conv_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype02+
)ecc_conv/FGN_out/Tensordot/ReadVariableOp
ecc_conv/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
ecc_conv/FGN_out/Tensordot/axes
ecc_conv/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
ecc_conv/FGN_out/Tensordot/free
 ecc_conv/FGN_out/Tensordot/ShapeShapeecc_conv/FGN_out/Cast:y:0*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/Shape
(ecc_conv/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/GatherV2/axis¦
#ecc_conv/FGN_out/Tensordot/GatherV2GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/free:output:01ecc_conv/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/GatherV2
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv/FGN_out/Tensordot/GatherV2_1/axis¬
%ecc_conv/FGN_out/Tensordot/GatherV2_1GatherV2)ecc_conv/FGN_out/Tensordot/Shape:output:0(ecc_conv/FGN_out/Tensordot/axes:output:03ecc_conv/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv/FGN_out/Tensordot/GatherV2_1
 ecc_conv/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ecc_conv/FGN_out/Tensordot/ConstÄ
ecc_conv/FGN_out/Tensordot/ProdProd,ecc_conv/FGN_out/Tensordot/GatherV2:output:0)ecc_conv/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
ecc_conv/FGN_out/Tensordot/Prod
"ecc_conv/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_1Ì
!ecc_conv/FGN_out/Tensordot/Prod_1Prod.ecc_conv/FGN_out/Tensordot/GatherV2_1:output:0+ecc_conv/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!ecc_conv/FGN_out/Tensordot/Prod_1
&ecc_conv/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&ecc_conv/FGN_out/Tensordot/concat/axis
!ecc_conv/FGN_out/Tensordot/concatConcatV2(ecc_conv/FGN_out/Tensordot/free:output:0(ecc_conv/FGN_out/Tensordot/axes:output:0/ecc_conv/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!ecc_conv/FGN_out/Tensordot/concatÐ
 ecc_conv/FGN_out/Tensordot/stackPack(ecc_conv/FGN_out/Tensordot/Prod:output:0*ecc_conv/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 ecc_conv/FGN_out/Tensordot/stackÚ
$ecc_conv/FGN_out/Tensordot/transpose	Transposeecc_conv/FGN_out/Cast:y:0*ecc_conv/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2&
$ecc_conv/FGN_out/Tensordot/transposeã
"ecc_conv/FGN_out/Tensordot/ReshapeReshape(ecc_conv/FGN_out/Tensordot/transpose:y:0)ecc_conv/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"ecc_conv/FGN_out/Tensordot/Reshapeã
!ecc_conv/FGN_out/Tensordot/MatMulMatMul+ecc_conv/FGN_out/Tensordot/Reshape:output:01ecc_conv/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ecc_conv/FGN_out/Tensordot/MatMul
"ecc_conv/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv/FGN_out/Tensordot/Const_2
(ecc_conv/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv/FGN_out/Tensordot/concat_1/axis
#ecc_conv/FGN_out/Tensordot/concat_1ConcatV2,ecc_conv/FGN_out/Tensordot/GatherV2:output:0+ecc_conv/FGN_out/Tensordot/Const_2:output:01ecc_conv/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv/FGN_out/Tensordot/concat_1Ù
ecc_conv/FGN_out/TensordotReshape+ecc_conv/FGN_out/Tensordot/MatMul:product:0,ecc_conv/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/TensordotÀ
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpReadVariableOp0ecc_conv_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02)
'ecc_conv/FGN_out/BiasAdd/ReadVariableOpÐ
ecc_conv/FGN_out/BiasAddBiasAdd#ecc_conv/FGN_out/Tensordot:output:0/ecc_conv/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
ecc_conv/FGN_out/BiasAdd
ecc_conv/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape/shape/0v
ecc_conv/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape/shape/3
ecc_conv/Reshape/shapePack!ecc_conv/Reshape/shape/0:output:0ecc_conv/strided_slice:output:0ecc_conv/strided_slice:output:0!ecc_conv/Reshape/shape/3:output:0!ecc_conv/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape/shapeÌ
ecc_conv/ReshapeReshape!ecc_conv/FGN_out/BiasAdd:output:0ecc_conv/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape
ecc_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
ecc_conv/strided_slice_2/stack
 ecc_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv/strided_slice_2/stack_1
 ecc_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 ecc_conv/strided_slice_2/stack_2Á
ecc_conv/strided_slice_2StridedSliceinput_2'ecc_conv/strided_slice_2/stack:output:0)ecc_conv/strided_slice_2/stack_1:output:0)ecc_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv/strided_slice_2¨
ecc_conv/mulMulecc_conv/Reshape:output:0!ecc_conv/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv/mulÏ
ecc_conv/einsum/EinsumEinsumecc_conv/mul:z:0$graph_masking/strided_slice:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv/einsum/Einsumx
ecc_conv/Shape_2Shape$graph_masking/strided_slice:output:0*
T0*
_output_shapes
:2
ecc_conv/Shape_2y
ecc_conv/unstackUnpackecc_conv/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv/unstack«
ecc_conv/Shape_3/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02!
ecc_conv/Shape_3/ReadVariableOpu
ecc_conv/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/Shape_3{
ecc_conv/unstack_1Unpackecc_conv/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv/unstack_1
ecc_conv/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
ecc_conv/Reshape_1/shape®
ecc_conv/Reshape_1Reshape$graph_masking/strided_slice:output:0!ecc_conv/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ecc_conv/Reshape_1¯
!ecc_conv/transpose/ReadVariableOpReadVariableOp(ecc_conv_shape_3_readvariableop_resource*
_output_shapes

: *
dtype02#
!ecc_conv/transpose/ReadVariableOp
ecc_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv/transpose/perm«
ecc_conv/transpose	Transpose)ecc_conv/transpose/ReadVariableOp:value:0 ecc_conv/transpose/perm:output:0*
T0*
_output_shapes

: 2
ecc_conv/transpose
ecc_conv/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
ecc_conv/Reshape_2/shape
ecc_conv/Reshape_2Reshapeecc_conv/transpose:y:0!ecc_conv/Reshape_2/shape:output:0*
T0*
_output_shapes

: 2
ecc_conv/Reshape_2
ecc_conv/MatMulMatMulecc_conv/Reshape_1:output:0ecc_conv/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv/MatMulz
ecc_conv/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv/Reshape_3/shape/1z
ecc_conv/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv/Reshape_3/shape/2Ï
ecc_conv/Reshape_3/shapePackecc_conv/unstack:output:0#ecc_conv/Reshape_3/shape/1:output:0#ecc_conv/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv/Reshape_3/shape§
ecc_conv/Reshape_3Reshapeecc_conv/MatMul:product:0!ecc_conv/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reshape_3
ecc_conv/addAddV2ecc_conv/einsum/Einsum:output:0ecc_conv/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/add§
ecc_conv/BiasAdd/ReadVariableOpReadVariableOp(ecc_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
ecc_conv/BiasAdd/ReadVariableOp 
ecc_conv/BiasAddBiasAddecc_conv/add:z:0'ecc_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/BiasAdd 
ecc_conv/mul_1Mulecc_conv/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/mul_1p
ecc_conv/ReluReluecc_conv/mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv/Reluo
ecc_conv_1/ShapeShapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape
ecc_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2 
ecc_conv_1/strided_slice/stack
 ecc_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice/stack_1
 ecc_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ecc_conv_1/strided_slice/stack_2¤
ecc_conv_1/strided_sliceStridedSliceecc_conv_1/Shape:output:0'ecc_conv_1/strided_slice/stack:output:0)ecc_conv_1/strided_slice/stack_1:output:0)ecc_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slices
ecc_conv_1/Shape_1Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_1
 ecc_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ecc_conv_1/strided_slice_1/stack
"ecc_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/strided_slice_1/stack_1
"ecc_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ecc_conv_1/strided_slice_1/stack_2°
ecc_conv_1/strided_slice_1StridedSliceecc_conv_1/Shape_1:output:0)ecc_conv_1/strided_slice_1/stack:output:0+ecc_conv_1/strided_slice_1/stack_1:output:0+ecc_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ecc_conv_1/strided_slice_1
ecc_conv_1/FGN_out/CastCastinput_3*

DstT0*

SrcT0	*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/CastÐ
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOpReadVariableOp4ecc_conv_1_fgn_out_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02-
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp
!ecc_conv_1/FGN_out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!ecc_conv_1/FGN_out/Tensordot/axes
!ecc_conv_1/FGN_out/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!ecc_conv_1/FGN_out/Tensordot/free
"ecc_conv_1/FGN_out/Tensordot/ShapeShapeecc_conv_1/FGN_out/Cast:y:0*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/Shape
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/GatherV2/axis°
%ecc_conv_1/FGN_out/Tensordot/GatherV2GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/free:output:03ecc_conv_1/FGN_out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/GatherV2
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis¶
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1GatherV2+ecc_conv_1/FGN_out/Tensordot/Shape:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:05ecc_conv_1/FGN_out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'ecc_conv_1/FGN_out/Tensordot/GatherV2_1
"ecc_conv_1/FGN_out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"ecc_conv_1/FGN_out/Tensordot/ConstÌ
!ecc_conv_1/FGN_out/Tensordot/ProdProd.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0+ecc_conv_1/FGN_out/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!ecc_conv_1/FGN_out/Tensordot/Prod
$ecc_conv_1/FGN_out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$ecc_conv_1/FGN_out/Tensordot/Const_1Ô
#ecc_conv_1/FGN_out/Tensordot/Prod_1Prod0ecc_conv_1/FGN_out/Tensordot/GatherV2_1:output:0-ecc_conv_1/FGN_out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#ecc_conv_1/FGN_out/Tensordot/Prod_1
(ecc_conv_1/FGN_out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(ecc_conv_1/FGN_out/Tensordot/concat/axis
#ecc_conv_1/FGN_out/Tensordot/concatConcatV2*ecc_conv_1/FGN_out/Tensordot/free:output:0*ecc_conv_1/FGN_out/Tensordot/axes:output:01ecc_conv_1/FGN_out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#ecc_conv_1/FGN_out/Tensordot/concatØ
"ecc_conv_1/FGN_out/Tensordot/stackPack*ecc_conv_1/FGN_out/Tensordot/Prod:output:0,ecc_conv_1/FGN_out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"ecc_conv_1/FGN_out/Tensordot/stackâ
&ecc_conv_1/FGN_out/Tensordot/transpose	Transposeecc_conv_1/FGN_out/Cast:y:0,ecc_conv_1/FGN_out/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2(
&ecc_conv_1/FGN_out/Tensordot/transposeë
$ecc_conv_1/FGN_out/Tensordot/ReshapeReshape*ecc_conv_1/FGN_out/Tensordot/transpose:y:0+ecc_conv_1/FGN_out/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$ecc_conv_1/FGN_out/Tensordot/Reshapeë
#ecc_conv_1/FGN_out/Tensordot/MatMulMatMul-ecc_conv_1/FGN_out/Tensordot/Reshape:output:03ecc_conv_1/FGN_out/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ecc_conv_1/FGN_out/Tensordot/MatMul
$ecc_conv_1/FGN_out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$ecc_conv_1/FGN_out/Tensordot/Const_2
*ecc_conv_1/FGN_out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*ecc_conv_1/FGN_out/Tensordot/concat_1/axis
%ecc_conv_1/FGN_out/Tensordot/concat_1ConcatV2.ecc_conv_1/FGN_out/Tensordot/GatherV2:output:0-ecc_conv_1/FGN_out/Tensordot/Const_2:output:03ecc_conv_1/FGN_out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%ecc_conv_1/FGN_out/Tensordot/concat_1á
ecc_conv_1/FGN_out/TensordotReshape-ecc_conv_1/FGN_out/Tensordot/MatMul:product:0.ecc_conv_1/FGN_out/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/TensordotÆ
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpReadVariableOp2ecc_conv_1_fgn_out_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOpØ
ecc_conv_1/FGN_out/BiasAddBiasAdd%ecc_conv_1/FGN_out/Tensordot:output:01ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

2
ecc_conv_1/FGN_out/BiasAdd
ecc_conv_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape/shape/0z
ecc_conv_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape/shape/3
ecc_conv_1/Reshape/shapePack#ecc_conv_1/Reshape/shape/0:output:0!ecc_conv_1/strided_slice:output:0!ecc_conv_1/strided_slice:output:0#ecc_conv_1/Reshape/shape/3:output:0#ecc_conv_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape/shapeÔ
ecc_conv_1/ReshapeReshape#ecc_conv_1/FGN_out/BiasAdd:output:0!ecc_conv_1/Reshape/shape:output:0*
T0*N
_output_shapes<
::8ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/Reshape
 ecc_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2"
 ecc_conv_1/strided_slice_2/stack
"ecc_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2$
"ecc_conv_1/strided_slice_2/stack_1
"ecc_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2$
"ecc_conv_1/strided_slice_2/stack_2Ë
ecc_conv_1/strided_slice_2StridedSliceinput_2)ecc_conv_1/strided_slice_2/stack:output:0+ecc_conv_1/strided_slice_2/stack_1:output:0+ecc_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ

*
ellipsis_mask*
new_axis_mask2
ecc_conv_1/strided_slice_2°
ecc_conv_1/mulMulecc_conv_1/Reshape:output:0#ecc_conv_1/strided_slice_2:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿ

 ÿÿÿÿÿÿÿÿÿ2
ecc_conv_1/mulÌ
ecc_conv_1/einsum/EinsumEinsumecc_conv_1/mul:z:0ecc_conv/Relu:activations:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
equationabcde,ace->abd2
ecc_conv_1/einsum/Einsums
ecc_conv_1/Shape_2Shapeecc_conv/Relu:activations:0*
T0*
_output_shapes
:2
ecc_conv_1/Shape_2
ecc_conv_1/unstackUnpackecc_conv_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
ecc_conv_1/unstack±
!ecc_conv_1/Shape_3/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02#
!ecc_conv_1/Shape_3/ReadVariableOpy
ecc_conv_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"        2
ecc_conv_1/Shape_3
ecc_conv_1/unstack_1Unpackecc_conv_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
ecc_conv_1/unstack_1
ecc_conv_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
ecc_conv_1/Reshape_1/shape«
ecc_conv_1/Reshape_1Reshapeecc_conv/Relu:activations:0#ecc_conv_1/Reshape_1/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/Reshape_1µ
#ecc_conv_1/transpose/ReadVariableOpReadVariableOp*ecc_conv_1_shape_3_readvariableop_resource*
_output_shapes

:  *
dtype02%
#ecc_conv_1/transpose/ReadVariableOp
ecc_conv_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
ecc_conv_1/transpose/perm³
ecc_conv_1/transpose	Transpose+ecc_conv_1/transpose/ReadVariableOp:value:0"ecc_conv_1/transpose/perm:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/transpose
ecc_conv_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
ecc_conv_1/Reshape_2/shape
ecc_conv_1/Reshape_2Reshapeecc_conv_1/transpose:y:0#ecc_conv_1/Reshape_2/shape:output:0*
T0*
_output_shapes

:  2
ecc_conv_1/Reshape_2 
ecc_conv_1/MatMulMatMulecc_conv_1/Reshape_1:output:0ecc_conv_1/Reshape_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ecc_conv_1/MatMul~
ecc_conv_1/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2
ecc_conv_1/Reshape_3/shape/1~
ecc_conv_1/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
ecc_conv_1/Reshape_3/shape/2Ù
ecc_conv_1/Reshape_3/shapePackecc_conv_1/unstack:output:0%ecc_conv_1/Reshape_3/shape/1:output:0%ecc_conv_1/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2
ecc_conv_1/Reshape_3/shape¯
ecc_conv_1/Reshape_3Reshapeecc_conv_1/MatMul:product:0#ecc_conv_1/Reshape_3/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/Reshape_3¡
ecc_conv_1/addAddV2!ecc_conv_1/einsum/Einsum:output:0ecc_conv_1/Reshape_3:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/add­
!ecc_conv_1/BiasAdd/ReadVariableOpReadVariableOp*ecc_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!ecc_conv_1/BiasAdd/ReadVariableOp¨
ecc_conv_1/BiasAddBiasAddecc_conv_1/add:z:0)ecc_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/BiasAdd¦
ecc_conv_1/mul_1Mulecc_conv_1/BiasAdd:output:0&graph_masking/strided_slice_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
ecc_conv_1/mul_1
%global_max_pool/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ2'
%global_max_pool/Max/reduction_indices©
global_max_pool/MaxMaxecc_conv_1/mul_1:z:0.global_max_pool/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
global_max_pool/Max 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulglobal_max_pool/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityË
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp ^ecc_conv/BiasAdd/ReadVariableOp(^ecc_conv/FGN_out/BiasAdd/ReadVariableOp*^ecc_conv/FGN_out/Tensordot/ReadVariableOp"^ecc_conv/transpose/ReadVariableOp"^ecc_conv_1/BiasAdd/ReadVariableOp*^ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp,^ecc_conv_1/FGN_out/Tensordot/ReadVariableOp$^ecc_conv_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2B
ecc_conv/BiasAdd/ReadVariableOpecc_conv/BiasAdd/ReadVariableOp2R
'ecc_conv/FGN_out/BiasAdd/ReadVariableOp'ecc_conv/FGN_out/BiasAdd/ReadVariableOp2V
)ecc_conv/FGN_out/Tensordot/ReadVariableOp)ecc_conv/FGN_out/Tensordot/ReadVariableOp2F
!ecc_conv/transpose/ReadVariableOp!ecc_conv/transpose/ReadVariableOp2F
!ecc_conv_1/BiasAdd/ReadVariableOp!ecc_conv_1/BiasAdd/ReadVariableOp2V
)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp)ecc_conv_1/FGN_out/BiasAdd/ReadVariableOp2Z
+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp+ecc_conv_1/FGN_out/Tensordot/ReadVariableOp2J
#ecc_conv_1/transpose/ReadVariableOp#ecc_conv_1/transpose/ReadVariableOp:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_2:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_3
Ú

$__inference_net_layer_call_fn_150866
input_1
input_2
input_3	
unknown:	 
	unknown_0:	 
	unknown_1: 
	unknown_2: 
	unknown_3:	
	unknown_4:	
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:	
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_net_layer_call_and_return_conditional_losses_1505262
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_2:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_3
¼

$__inference_signature_wrapper_150839
input_1
input_2
input_3	
unknown:	 
	unknown_0:	 
	unknown_1: 
	unknown_2: 
	unknown_3:	
	unknown_4:	
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:	
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1502942
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_2:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_3
¡
e
I__inference_graph_masking_layer_call_and_return_conditional_losses_150313

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ÿÿÿÿ2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*

begin_mask*
ellipsis_mask2
strided_slicen
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ú

$__inference_net_layer_call_fn_150947
input_1
input_2
input_3	
unknown:	 
	unknown_0:	 
	unknown_1: 
	unknown_2: 
	unknown_3:	
	unknown_4:	
	unknown_5:  
	unknown_6: 
	unknown_7:	 
	unknown_8:	
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_net_layer_call_and_return_conditional_losses_1506842
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ

:ÿÿÿÿÿÿÿÿÿ

: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1:TP
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_2:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


!
_user_specified_name	input_3
ý
g
K__inference_global_max_pool_layer_call_and_return_conditional_losses_150506

inputs
identityy
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
þÿÿÿÿÿÿÿÿ2
Max/reduction_indicesk
MaxMaxinputsMax/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Max`
IdentityIdentityMax:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
ø¡
¶
"__inference__traced_restore_152134
file_prefix;
)assignvariableop_net_ecc_conv_root_kernel: 2
$assignvariableop_1_net_ecc_conv_bias: ?
-assignvariableop_2_net_ecc_conv_1_root_kernel:  4
&assignvariableop_3_net_ecc_conv_1_bias: 6
#assignvariableop_4_net_dense_kernel:	 0
!assignvariableop_5_net_dense_bias:	&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: B
/assignvariableop_11_net_ecc_conv_fgn_out_kernel:	 <
-assignvariableop_12_net_ecc_conv_fgn_out_bias:	 D
1assignvariableop_13_net_ecc_conv_1_fgn_out_kernel:	>
/assignvariableop_14_net_ecc_conv_1_fgn_out_bias:	#
assignvariableop_15_total: #
assignvariableop_16_count: E
3assignvariableop_17_adam_net_ecc_conv_root_kernel_m: :
,assignvariableop_18_adam_net_ecc_conv_bias_m: G
5assignvariableop_19_adam_net_ecc_conv_1_root_kernel_m:  <
.assignvariableop_20_adam_net_ecc_conv_1_bias_m: >
+assignvariableop_21_adam_net_dense_kernel_m:	 8
)assignvariableop_22_adam_net_dense_bias_m:	I
6assignvariableop_23_adam_net_ecc_conv_fgn_out_kernel_m:	 C
4assignvariableop_24_adam_net_ecc_conv_fgn_out_bias_m:	 K
8assignvariableop_25_adam_net_ecc_conv_1_fgn_out_kernel_m:	E
6assignvariableop_26_adam_net_ecc_conv_1_fgn_out_bias_m:	E
3assignvariableop_27_adam_net_ecc_conv_root_kernel_v: :
,assignvariableop_28_adam_net_ecc_conv_bias_v: G
5assignvariableop_29_adam_net_ecc_conv_1_root_kernel_v:  <
.assignvariableop_30_adam_net_ecc_conv_1_bias_v: >
+assignvariableop_31_adam_net_dense_kernel_v:	 8
)assignvariableop_32_adam_net_dense_bias_v:	I
6assignvariableop_33_adam_net_ecc_conv_fgn_out_kernel_v:	 C
4assignvariableop_34_adam_net_ecc_conv_fgn_out_bias_v:	 K
8assignvariableop_35_adam_net_ecc_conv_1_fgn_out_kernel_v:	E
6assignvariableop_36_adam_net_ecc_conv_1_fgn_out_bias_v:	
identity_38¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9î
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ú
valueðBí&B,conv1/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB,conv2/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBHconv1/root_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconv2/root_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconv1/root_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconv2/root_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÚ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¨
AssignVariableOpAssignVariableOp)assignvariableop_net_ecc_conv_root_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1©
AssignVariableOp_1AssignVariableOp$assignvariableop_1_net_ecc_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2²
AssignVariableOp_2AssignVariableOp-assignvariableop_2_net_ecc_conv_1_root_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3«
AssignVariableOp_3AssignVariableOp&assignvariableop_3_net_ecc_conv_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_net_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_net_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6¡
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_net_ecc_conv_fgn_out_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12µ
AssignVariableOp_12AssignVariableOp-assignvariableop_12_net_ecc_conv_fgn_out_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¹
AssignVariableOp_13AssignVariableOp1assignvariableop_13_net_ecc_conv_1_fgn_out_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14·
AssignVariableOp_14AssignVariableOp/assignvariableop_14_net_ecc_conv_1_fgn_out_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¡
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17»
AssignVariableOp_17AssignVariableOp3assignvariableop_17_adam_net_ecc_conv_root_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18´
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_net_ecc_conv_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19½
AssignVariableOp_19AssignVariableOp5assignvariableop_19_adam_net_ecc_conv_1_root_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¶
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_net_ecc_conv_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_net_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_net_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¾
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_net_ecc_conv_fgn_out_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¼
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_net_ecc_conv_fgn_out_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_net_ecc_conv_1_fgn_out_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_net_ecc_conv_1_fgn_out_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27»
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_net_ecc_conv_root_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28´
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_net_ecc_conv_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29½
AssignVariableOp_29AssignVariableOp5assignvariableop_29_adam_net_ecc_conv_1_root_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¶
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_net_ecc_conv_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_net_dense_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_net_dense_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¾
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_net_ecc_conv_fgn_out_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¼
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_net_ecc_conv_fgn_out_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35À
AssignVariableOp_35AssignVariableOp8assignvariableop_35_adam_net_ecc_conv_1_fgn_out_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¾
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_net_ecc_conv_1_fgn_out_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37f
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_38ô
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ö
J
.__inference_graph_masking_layer_call_fn_151648

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_graph_masking_layer_call_and_return_conditional_losses_1503132
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
íQ
¢
__inference__traced_save_152013
file_prefix7
3savev2_net_ecc_conv_root_kernel_read_readvariableop0
,savev2_net_ecc_conv_bias_read_readvariableop9
5savev2_net_ecc_conv_1_root_kernel_read_readvariableop2
.savev2_net_ecc_conv_1_bias_read_readvariableop/
+savev2_net_dense_kernel_read_readvariableop-
)savev2_net_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_net_ecc_conv_fgn_out_kernel_read_readvariableop8
4savev2_net_ecc_conv_fgn_out_bias_read_readvariableop<
8savev2_net_ecc_conv_1_fgn_out_kernel_read_readvariableop:
6savev2_net_ecc_conv_1_fgn_out_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_net_ecc_conv_root_kernel_m_read_readvariableop7
3savev2_adam_net_ecc_conv_bias_m_read_readvariableop@
<savev2_adam_net_ecc_conv_1_root_kernel_m_read_readvariableop9
5savev2_adam_net_ecc_conv_1_bias_m_read_readvariableop6
2savev2_adam_net_dense_kernel_m_read_readvariableop4
0savev2_adam_net_dense_bias_m_read_readvariableopA
=savev2_adam_net_ecc_conv_fgn_out_kernel_m_read_readvariableop?
;savev2_adam_net_ecc_conv_fgn_out_bias_m_read_readvariableopC
?savev2_adam_net_ecc_conv_1_fgn_out_kernel_m_read_readvariableopA
=savev2_adam_net_ecc_conv_1_fgn_out_bias_m_read_readvariableop>
:savev2_adam_net_ecc_conv_root_kernel_v_read_readvariableop7
3savev2_adam_net_ecc_conv_bias_v_read_readvariableop@
<savev2_adam_net_ecc_conv_1_root_kernel_v_read_readvariableop9
5savev2_adam_net_ecc_conv_1_bias_v_read_readvariableop6
2savev2_adam_net_dense_kernel_v_read_readvariableop4
0savev2_adam_net_dense_bias_v_read_readvariableopA
=savev2_adam_net_ecc_conv_fgn_out_kernel_v_read_readvariableop?
;savev2_adam_net_ecc_conv_fgn_out_bias_v_read_readvariableopC
?savev2_adam_net_ecc_conv_1_fgn_out_kernel_v_read_readvariableopA
=savev2_adam_net_ecc_conv_1_fgn_out_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameè
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ú
valueðBí&B,conv1/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB,conv2/root_kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBHconv1/root_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconv2/root_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHconv1/root_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHconv2/root_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÔ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_net_ecc_conv_root_kernel_read_readvariableop,savev2_net_ecc_conv_bias_read_readvariableop5savev2_net_ecc_conv_1_root_kernel_read_readvariableop.savev2_net_ecc_conv_1_bias_read_readvariableop+savev2_net_dense_kernel_read_readvariableop)savev2_net_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_net_ecc_conv_fgn_out_kernel_read_readvariableop4savev2_net_ecc_conv_fgn_out_bias_read_readvariableop8savev2_net_ecc_conv_1_fgn_out_kernel_read_readvariableop6savev2_net_ecc_conv_1_fgn_out_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_net_ecc_conv_root_kernel_m_read_readvariableop3savev2_adam_net_ecc_conv_bias_m_read_readvariableop<savev2_adam_net_ecc_conv_1_root_kernel_m_read_readvariableop5savev2_adam_net_ecc_conv_1_bias_m_read_readvariableop2savev2_adam_net_dense_kernel_m_read_readvariableop0savev2_adam_net_dense_bias_m_read_readvariableop=savev2_adam_net_ecc_conv_fgn_out_kernel_m_read_readvariableop;savev2_adam_net_ecc_conv_fgn_out_bias_m_read_readvariableop?savev2_adam_net_ecc_conv_1_fgn_out_kernel_m_read_readvariableop=savev2_adam_net_ecc_conv_1_fgn_out_bias_m_read_readvariableop:savev2_adam_net_ecc_conv_root_kernel_v_read_readvariableop3savev2_adam_net_ecc_conv_bias_v_read_readvariableop<savev2_adam_net_ecc_conv_1_root_kernel_v_read_readvariableop5savev2_adam_net_ecc_conv_1_bias_v_read_readvariableop2savev2_adam_net_dense_kernel_v_read_readvariableop0savev2_adam_net_dense_bias_v_read_readvariableop=savev2_adam_net_ecc_conv_fgn_out_kernel_v_read_readvariableop;savev2_adam_net_ecc_conv_fgn_out_bias_v_read_readvariableop?savev2_adam_net_ecc_conv_1_fgn_out_kernel_v_read_readvariableop=savev2_adam_net_ecc_conv_1_fgn_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*©
_input_shapes
: : : :  : :	 :: : : : : :	 : :	:: : : : :  : :	 ::	 : :	:: : :  : :	 ::	 : :	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
: :%!

_output_shapes
:	:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :%!

_output_shapes
:	 :!

_output_shapes	
::%!

_output_shapes
:	 :!

_output_shapes	
: :%!

_output_shapes
:	:!

_output_shapes	
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :% !

_output_shapes
:	 :!!

_output_shapes	
::%"!

_output_shapes
:	 :!#

_output_shapes	
: :%$!

_output_shapes
:	:!%

_output_shapes	
::&

_output_shapes
: "¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¶
serving_default¢
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ

?
input_24
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ


C
input_38
serving_default_input_3:0	ÿÿÿÿÿÿÿÿÿ

=
output_11
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÿ
Á
masking
	conv1
	conv2
global_pool1
global_pool2
	activ
	dense
	optimizer
	trainable_variables

	variables
regularization_losses
	keras_api

signatures
_default_save_signature
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_model
§
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
î
kwargs_keys
kernel_network_layers
root_kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
î
kwargs_keys
kernel_network_layers
root_kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
§
"trainable_variables
#	variables
$regularization_losses
%	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
(
&	keras_api"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
½

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer

.iter

/beta_1

0beta_2
	1decay
2learning_ratemnmompmq(mr)ms3mt4mu5mv6mwvxvyvzv{(v|)v}3v~4v5v6v"
	optimizer
f
0
1
32
43
4
5
56
67
(8
)9"
trackable_list_wrapper
f
0
1
32
43
4
5
56
67
(8
)9"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
	trainable_variables
7layer_metrics
8metrics

	variables
9layer_regularization_losses
regularization_losses

:layers
;non_trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
<layer_metrics
=metrics
>layer_regularization_losses
	variables
regularization_losses

?layers
@non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
*:( 2net/ecc_conv/root_kernel
: 2net/ecc_conv/bias
<
0
1
32
43"
trackable_list_wrapper
<
0
1
32
43"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
Blayer_metrics
Cmetrics
Dlayer_regularization_losses
	variables
regularization_losses

Elayers
Fnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
,:*  2net/ecc_conv_1/root_kernel
!: 2net/ecc_conv_1/bias
<
0
1
52
63"
trackable_list_wrapper
<
0
1
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
Hlayer_metrics
Imetrics
Jlayer_regularization_losses
	variables
 regularization_losses

Klayers
Lnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
"trainable_variables
Mlayer_metrics
Nmetrics
Olayer_regularization_losses
#	variables
$regularization_losses

Players
Qnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!	 2net/dense/kernel
:2net/dense/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
*trainable_variables
Rlayer_metrics
Smetrics
Tlayer_regularization_losses
+	variables
,regularization_losses

Ulayers
Vnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	 2net/ecc_conv/FGN_out/kernel
(:& 2net/ecc_conv/FGN_out/bias
0:.	2net/ecc_conv_1/FGN_out/kernel
*:(2net/ecc_conv_1/FGN_out/bias
 "
trackable_dict_wrapper
'
W0"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
½

3kernel
4bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
½

5kernel
6bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	`total
	acount
b	variables
c	keras_api"
_tf_keras_metric
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Xtrainable_variables
dlayer_metrics
emetrics
flayer_regularization_losses
Y	variables
Zregularization_losses

glayers
hnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
°
\trainable_variables
ilayer_metrics
jmetrics
klayer_regularization_losses
]	variables
^regularization_losses

llayers
mnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
.
`0
a1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/:- 2Adam/net/ecc_conv/root_kernel/m
$:" 2Adam/net/ecc_conv/bias/m
1:/  2!Adam/net/ecc_conv_1/root_kernel/m
&:$ 2Adam/net/ecc_conv_1/bias/m
(:&	 2Adam/net/dense/kernel/m
": 2Adam/net/dense/bias/m
3:1	 2"Adam/net/ecc_conv/FGN_out/kernel/m
-:+ 2 Adam/net/ecc_conv/FGN_out/bias/m
5:3	2$Adam/net/ecc_conv_1/FGN_out/kernel/m
/:-2"Adam/net/ecc_conv_1/FGN_out/bias/m
/:- 2Adam/net/ecc_conv/root_kernel/v
$:" 2Adam/net/ecc_conv/bias/v
1:/  2!Adam/net/ecc_conv_1/root_kernel/v
&:$ 2Adam/net/ecc_conv_1/bias/v
(:&	 2Adam/net/dense/kernel/v
": 2Adam/net/dense/bias/v
3:1	 2"Adam/net/ecc_conv/FGN_out/kernel/v
-:+ 2 Adam/net/ecc_conv/FGN_out/bias/v
5:3	2$Adam/net/ecc_conv_1/FGN_out/kernel/v
/:-2"Adam/net/ecc_conv_1/FGN_out/bias/v
ÞBÛ
!__inference__wrapped_model_150294input_1input_2input_3"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
$__inference_net_layer_call_fn_150866
$__inference_net_layer_call_fn_150893
$__inference_net_layer_call_fn_150920
$__inference_net_layer_call_fn_150947³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
½2º
?__inference_net_layer_call_and_return_conditional_losses_151117
?__inference_net_layer_call_and_return_conditional_losses_151287
?__inference_net_layer_call_and_return_conditional_losses_151457
?__inference_net_layer_call_and_return_conditional_losses_151627³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
I__inference_graph_masking_layer_call_and_return_conditional_losses_151635
I__inference_graph_masking_layer_call_and_return_conditional_losses_151643Æ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
¬2©
.__inference_graph_masking_layer_call_fn_151648
.__inference_graph_masking_layer_call_fn_151653Æ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
ê2ç
D__inference_ecc_conv_layer_call_and_return_conditional_losses_151734
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
)__inference_ecc_conv_layer_call_fn_151750
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
F__inference_ecc_conv_1_layer_call_and_return_conditional_losses_151830
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
+__inference_ecc_conv_1_layer_call_fn_151846
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_global_max_pool_layer_call_and_return_conditional_losses_151852¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_global_max_pool_layer_call_fn_151857¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_151868¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_151877¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÛBØ
$__inference_signature_wrapper_150839input_1input_2input_3"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ø
!__inference__wrapped_model_150294Ò
3456()¢
¢~
|¢y
%"
input_1ÿÿÿÿÿÿÿÿÿ

%"
input_2ÿÿÿÿÿÿÿÿÿ


)&
input_3ÿÿÿÿÿÿÿÿÿ

	
ª "4ª1
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿ¢
A__inference_dense_layer_call_and_return_conditional_losses_151868]()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
&__inference_dense_layer_call_fn_151877P()/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿË
F__inference_ecc_conv_1_layer_call_and_return_conditional_losses_15183056Ì¢È
¢
|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
>ª;
9
mask1.
$!
mask/0ÿÿÿÿÿÿÿÿÿ


 

 ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 £
+__inference_ecc_conv_1_layer_call_fn_151846ó56Ì¢È
¢
|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
>ª;
9
mask1.
$!
mask/0ÿÿÿÿÿÿÿÿÿ


 

 "ÿÿÿÿÿÿÿÿÿ
 É
D__inference_ecc_conv_layer_call_and_return_conditional_losses_15173434Ì¢È
¢
|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
>ª;
9
mask1.
$!
mask/0ÿÿÿÿÿÿÿÿÿ


 

 ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 ¡
)__inference_ecc_conv_layer_call_fn_151750ó34Ì¢È
¢
|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
>ª;
9
mask1.
$!
mask/0ÿÿÿÿÿÿÿÿÿ


 

 "ÿÿÿÿÿÿÿÿÿ
 «
K__inference_global_max_pool_layer_call_and_return_conditional_losses_151852\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_global_max_pool_layer_call_fn_151857O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
 
ª "ÿÿÿÿÿÿÿÿÿ ½
I__inference_graph_masking_layer_call_and_return_conditional_losses_151635pC¢@
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp ")¢&

0ÿÿÿÿÿÿÿÿÿ

 ½
I__inference_graph_masking_layer_call_and_return_conditional_losses_151643pC¢@
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp")¢&

0ÿÿÿÿÿÿÿÿÿ

 
.__inference_graph_masking_layer_call_fn_151648cC¢@
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp "ÿÿÿÿÿÿÿÿÿ

.__inference_graph_masking_layer_call_fn_151653cC¢@
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp"ÿÿÿÿÿÿÿÿÿ

?__inference_net_layer_call_and_return_conditional_losses_151117Ì
3456()¢
¢
¢|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
?__inference_net_layer_call_and_return_conditional_losses_151287Ì
3456()¢
¢
¢|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
?__inference_net_layer_call_and_return_conditional_losses_151457É
3456()¢
¢
|¢y
%"
input_1ÿÿÿÿÿÿÿÿÿ

%"
input_2ÿÿÿÿÿÿÿÿÿ


)&
input_3ÿÿÿÿÿÿÿÿÿ

	
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
?__inference_net_layer_call_and_return_conditional_losses_151627É
3456()¢
¢
|¢y
%"
input_1ÿÿÿÿÿÿÿÿÿ

%"
input_2ÿÿÿÿÿÿÿÿÿ


)&
input_3ÿÿÿÿÿÿÿÿÿ

	
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 å
$__inference_net_layer_call_fn_150866¼
3456()¢
¢
|¢y
%"
input_1ÿÿÿÿÿÿÿÿÿ

%"
input_2ÿÿÿÿÿÿÿÿÿ


)&
input_3ÿÿÿÿÿÿÿÿÿ

	
p 
ª "ÿÿÿÿÿÿÿÿÿè
$__inference_net_layer_call_fn_150893¿
3456()¢
¢
¢|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
p 
ª "ÿÿÿÿÿÿÿÿÿè
$__inference_net_layer_call_fn_150920¿
3456()¢
¢
¢|
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

&#
inputs/1ÿÿÿÿÿÿÿÿÿ


*'
inputs/2ÿÿÿÿÿÿÿÿÿ

	
p
ª "ÿÿÿÿÿÿÿÿÿå
$__inference_net_layer_call_fn_150947¼
3456()¢
¢
|¢y
%"
input_1ÿÿÿÿÿÿÿÿÿ

%"
input_2ÿÿÿÿÿÿÿÿÿ


)&
input_3ÿÿÿÿÿÿÿÿÿ

	
p
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_150839ï
3456()ª¢¦
¢ 
ª
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿ

0
input_2%"
input_2ÿÿÿÿÿÿÿÿÿ


4
input_3)&
input_3ÿÿÿÿÿÿÿÿÿ

	"4ª1
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿ