щФ
яг
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЭЬL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

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
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
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
 "serve*2.9.12unknown8­
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

DMDense_OUT/DMDense_OUT_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameDMDense_OUT/DMDense_OUT_bias

0DMDense_OUT/DMDense_OUT_bias/Read/ReadVariableOpReadVariableOpDMDense_OUT/DMDense_OUT_bias*
_output_shapes
:*
dtype0

DMDense_OUT/DMDense_OUT_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name DMDense_OUT/DMDense_OUT_weight

2DMDense_OUT/DMDense_OUT_weight/Read/ReadVariableOpReadVariableOpDMDense_OUT/DMDense_OUT_weight*
_output_shapes

:*
dtype0
Є
&DMDense_Hidden_1/DMDense_Hidden_1_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&DMDense_Hidden_1/DMDense_Hidden_1_bias

:DMDense_Hidden_1/DMDense_Hidden_1_bias/Read/ReadVariableOpReadVariableOp&DMDense_Hidden_1/DMDense_Hidden_1_bias*
_output_shapes
:*
dtype0
Ќ
(DMDense_Hidden_1/DMDense_Hidden_1_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(DMDense_Hidden_1/DMDense_Hidden_1_weight
Ѕ
<DMDense_Hidden_1/DMDense_Hidden_1_weight/Read/ReadVariableOpReadVariableOp(DMDense_Hidden_1/DMDense_Hidden_1_weight*
_output_shapes

:*
dtype0
Є
&DMDense_Hidden_0/DMDense_Hidden_0_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&DMDense_Hidden_0/DMDense_Hidden_0_bias

:DMDense_Hidden_0/DMDense_Hidden_0_bias/Read/ReadVariableOpReadVariableOp&DMDense_Hidden_0/DMDense_Hidden_0_bias*
_output_shapes
:*
dtype0
Ќ
(DMDense_Hidden_0/DMDense_Hidden_0_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *9
shared_name*(DMDense_Hidden_0/DMDense_Hidden_0_weight
Ѕ
<DMDense_Hidden_0/DMDense_Hidden_0_weight/Read/ReadVariableOpReadVariableOp(DMDense_Hidden_0/DMDense_Hidden_0_weight*
_output_shapes

: *
dtype0
 
$DMGAttention_1/DMGAttention_1_1_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$DMGAttention_1/DMGAttention_1_1_bias

8DMGAttention_1/DMGAttention_1_1_bias/Read/ReadVariableOpReadVariableOp$DMGAttention_1/DMGAttention_1_1_bias*
_output_shapes
: *
dtype0
Ю
9DMGAttention_1/DMGAttention_1_1_neighbor_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *J
shared_name;9DMGAttention_1/DMGAttention_1_1_neighbor_attention_weight
Ч
MDMGAttention_1/DMGAttention_1_1_neighbor_attention_weight/Read/ReadVariableOpReadVariableOp9DMGAttention_1/DMGAttention_1_1_neighbor_attention_weight*
_output_shapes

: *
dtype0
Ц
5DMGAttention_1/DMGAttention_1_1_self_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75DMGAttention_1/DMGAttention_1_1_self_attention_weight
П
IDMGAttention_1/DMGAttention_1_1_self_attention_weight/Read/ReadVariableOpReadVariableOp5DMGAttention_1/DMGAttention_1_1_self_attention_weight*
_output_shapes

: *
dtype0
Ј
&DMGAttention_1/DMGAttention_1_1_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *7
shared_name(&DMGAttention_1/DMGAttention_1_1_weight
Ё
:DMGAttention_1/DMGAttention_1_1_weight/Read/ReadVariableOpReadVariableOp&DMGAttention_1/DMGAttention_1_1_weight*
_output_shapes

:  *
dtype0
 
$DMGAttention_1/DMGAttention_1_0_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$DMGAttention_1/DMGAttention_1_0_bias

8DMGAttention_1/DMGAttention_1_0_bias/Read/ReadVariableOpReadVariableOp$DMGAttention_1/DMGAttention_1_0_bias*
_output_shapes
: *
dtype0
Ю
9DMGAttention_1/DMGAttention_1_0_neighbor_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *J
shared_name;9DMGAttention_1/DMGAttention_1_0_neighbor_attention_weight
Ч
MDMGAttention_1/DMGAttention_1_0_neighbor_attention_weight/Read/ReadVariableOpReadVariableOp9DMGAttention_1/DMGAttention_1_0_neighbor_attention_weight*
_output_shapes

: *
dtype0
Ц
5DMGAttention_1/DMGAttention_1_0_self_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75DMGAttention_1/DMGAttention_1_0_self_attention_weight
П
IDMGAttention_1/DMGAttention_1_0_self_attention_weight/Read/ReadVariableOpReadVariableOp5DMGAttention_1/DMGAttention_1_0_self_attention_weight*
_output_shapes

: *
dtype0
Ј
&DMGAttention_1/DMGAttention_1_0_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *7
shared_name(&DMGAttention_1/DMGAttention_1_0_weight
Ё
:DMGAttention_1/DMGAttention_1_0_weight/Read/ReadVariableOpReadVariableOp&DMGAttention_1/DMGAttention_1_0_weight*
_output_shapes

:  *
dtype0
 
$DMGAttention_0/DMGAttention_0_1_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$DMGAttention_0/DMGAttention_0_1_bias

8DMGAttention_0/DMGAttention_0_1_bias/Read/ReadVariableOpReadVariableOp$DMGAttention_0/DMGAttention_0_1_bias*
_output_shapes
: *
dtype0
Ю
9DMGAttention_0/DMGAttention_0_1_neighbor_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *J
shared_name;9DMGAttention_0/DMGAttention_0_1_neighbor_attention_weight
Ч
MDMGAttention_0/DMGAttention_0_1_neighbor_attention_weight/Read/ReadVariableOpReadVariableOp9DMGAttention_0/DMGAttention_0_1_neighbor_attention_weight*
_output_shapes

: *
dtype0
Ц
5DMGAttention_0/DMGAttention_0_1_self_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75DMGAttention_0/DMGAttention_0_1_self_attention_weight
П
IDMGAttention_0/DMGAttention_0_1_self_attention_weight/Read/ReadVariableOpReadVariableOp5DMGAttention_0/DMGAttention_0_1_self_attention_weight*
_output_shapes

: *
dtype0
Ј
&DMGAttention_0/DMGAttention_0_1_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *7
shared_name(&DMGAttention_0/DMGAttention_0_1_weight
Ё
:DMGAttention_0/DMGAttention_0_1_weight/Read/ReadVariableOpReadVariableOp&DMGAttention_0/DMGAttention_0_1_weight*
_output_shapes

:( *
dtype0
 
$DMGAttention_0/DMGAttention_0_0_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$DMGAttention_0/DMGAttention_0_0_bias

8DMGAttention_0/DMGAttention_0_0_bias/Read/ReadVariableOpReadVariableOp$DMGAttention_0/DMGAttention_0_0_bias*
_output_shapes
: *
dtype0
Ю
9DMGAttention_0/DMGAttention_0_0_neighbor_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *J
shared_name;9DMGAttention_0/DMGAttention_0_0_neighbor_attention_weight
Ч
MDMGAttention_0/DMGAttention_0_0_neighbor_attention_weight/Read/ReadVariableOpReadVariableOp9DMGAttention_0/DMGAttention_0_0_neighbor_attention_weight*
_output_shapes

: *
dtype0
Ц
5DMGAttention_0/DMGAttention_0_0_self_attention_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75DMGAttention_0/DMGAttention_0_0_self_attention_weight
П
IDMGAttention_0/DMGAttention_0_0_self_attention_weight/Read/ReadVariableOpReadVariableOp5DMGAttention_0/DMGAttention_0_0_self_attention_weight*
_output_shapes

: *
dtype0
Ј
&DMGAttention_0/DMGAttention_0_0_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *7
shared_name(&DMGAttention_0/DMGAttention_0_0_weight
Ё
:DMGAttention_0/DMGAttention_0_0_weight/Read/ReadVariableOpReadVariableOp&DMGAttention_0/DMGAttention_0_0_weight*
_output_shapes

:( *
dtype0

NoOpNoOp
З\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ђ[
valueш[Bх[ Bо[
Ї
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
w
self_attention_w
neighbor_attention_w
bias
DMGAttention_0_0_weight
*&DMGAttention_0_0_self_attention_weight
.*DMGAttention_0_0_neighbor_attention_weight
DMGAttention_0_0_bias
DMGAttention_0_1_weight
* &DMGAttention_0_1_self_attention_weight
.!*DMGAttention_0_1_neighbor_attention_weight
"DMGAttention_0_1_bias
#attention_dropout
$feature_dropout*
Ѕ
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+w
,self_attention_w
-neighbor_attention_w
.bias
/DMGAttention_1_0_weight
*0&DMGAttention_1_0_self_attention_weight
.1*DMGAttention_1_0_neighbor_attention_weight
2DMGAttention_1_0_bias
3DMGAttention_1_1_weight
*4&DMGAttention_1_1_self_attention_weight
.5*DMGAttention_1_1_neighbor_attention_weight
6DMGAttention_1_1_bias
7attention_dropout
8feature_dropout*

9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses* 
о
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
EDMDense_Hidden_0_weight

Eweight
FDMDense_Hidden_0_bias
Fbias*
о
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
MDMDense_Hidden_1_weight

Mweight
NDMDense_Hidden_1_bias
Nbias*
д
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
UDMDense_OUT_weight

Uweight
VDMDense_OUT_bias
Vbias*
Њ
0
1
2
3
4
 5
!6
"7
/8
09
110
211
312
413
514
615
E16
F17
M18
N19
U20
V21*
Њ
0
1
2
3
4
 5
!6
"7
/8
09
110
211
312
413
514
615
E16
F17
M18
N19
U20
V21*
* 
А
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
\trace_0
]trace_1
^trace_2
_trace_3* 
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
* 

dserving_default* 
<
0
1
2
3
4
 5
!6
"7*
<
0
1
2
3
4
 5
!6
"7*
* 

enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

jtrace_0
ktrace_1* 

ltrace_0
mtrace_1* 

0
1*

0
 1*

0
!1*

0
"1*

VARIABLE_VALUE&DMGAttention_0/DMGAttention_0_0_weightGlayer_with_weights-0/DMGAttention_0_0_weight/.ATTRIBUTES/VARIABLE_VALUE*
І
VARIABLE_VALUE5DMGAttention_0/DMGAttention_0_0_self_attention_weightVlayer_with_weights-0/DMGAttention_0_0_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
ЎЇ
VARIABLE_VALUE9DMGAttention_0/DMGAttention_0_0_neighbor_attention_weightZlayer_with_weights-0/DMGAttention_0_0_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE$DMGAttention_0/DMGAttention_0_0_biasElayer_with_weights-0/DMGAttention_0_0_bias/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&DMGAttention_0/DMGAttention_0_1_weightGlayer_with_weights-0/DMGAttention_0_1_weight/.ATTRIBUTES/VARIABLE_VALUE*
І
VARIABLE_VALUE5DMGAttention_0/DMGAttention_0_1_self_attention_weightVlayer_with_weights-0/DMGAttention_0_1_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
ЎЇ
VARIABLE_VALUE9DMGAttention_0/DMGAttention_0_1_neighbor_attention_weightZlayer_with_weights-0/DMGAttention_0_1_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE$DMGAttention_0/DMGAttention_0_1_biasElayer_with_weights-0/DMGAttention_0_1_bias/.ATTRIBUTES/VARIABLE_VALUE*

n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 

t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 
<
/0
01
12
23
34
45
56
67*
<
/0
01
12
23
34
45
56
67*
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 

/0
31*

00
41*

10
51*

20
61*

VARIABLE_VALUE&DMGAttention_1/DMGAttention_1_0_weightGlayer_with_weights-1/DMGAttention_1_0_weight/.ATTRIBUTES/VARIABLE_VALUE*
І
VARIABLE_VALUE5DMGAttention_1/DMGAttention_1_0_self_attention_weightVlayer_with_weights-1/DMGAttention_1_0_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
ЎЇ
VARIABLE_VALUE9DMGAttention_1/DMGAttention_1_0_neighbor_attention_weightZlayer_with_weights-1/DMGAttention_1_0_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE$DMGAttention_1/DMGAttention_1_0_biasElayer_with_weights-1/DMGAttention_1_0_bias/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&DMGAttention_1/DMGAttention_1_1_weightGlayer_with_weights-1/DMGAttention_1_1_weight/.ATTRIBUTES/VARIABLE_VALUE*
І
VARIABLE_VALUE5DMGAttention_1/DMGAttention_1_1_self_attention_weightVlayer_with_weights-1/DMGAttention_1_1_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
ЎЇ
VARIABLE_VALUE9DMGAttention_1/DMGAttention_1_1_neighbor_attention_weightZlayer_with_weights-1/DMGAttention_1_1_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE$DMGAttention_1/DMGAttention_1_1_biasElayer_with_weights-1/DMGAttention_1_1_bias/.ATTRIBUTES/VARIABLE_VALUE*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

E0
F1*

E0
F1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

trace_0* 

trace_0* 

VARIABLE_VALUE(DMDense_Hidden_0/DMDense_Hidden_0_weightGlayer_with_weights-2/DMDense_Hidden_0_weight/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&DMDense_Hidden_0/DMDense_Hidden_0_biasElayer_with_weights-2/DMDense_Hidden_0_bias/.ATTRIBUTES/VARIABLE_VALUE*

M0
N1*

M0
N1*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

Ђtrace_0* 

Ѓtrace_0* 

VARIABLE_VALUE(DMDense_Hidden_1/DMDense_Hidden_1_weightGlayer_with_weights-3/DMDense_Hidden_1_weight/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&DMDense_Hidden_1/DMDense_Hidden_1_biasElayer_with_weights-3/DMDense_Hidden_1_bias/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

U0
V1*
* 

Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

Љtrace_0* 

Њtrace_0* 
zt
VARIABLE_VALUEDMDense_OUT/DMDense_OUT_weightBlayer_with_weights-4/DMDense_OUT_weight/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEDMDense_OUT/DMDense_OUT_bias@layer_with_weights-4/DMDense_OUT_bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

Ћ0
Ќ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

#0
$1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 
* 
* 
* 

70
81* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
С	variables
Т	keras_api

Уtotal

Фcount*
M
Х	variables
Ц	keras_api

Чtotal

Шcount
Щ
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

У0
Ф1*

С	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ч0
Ш1*

Х	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Џ
 serving_default_Adjacency_MatrixPlaceholder*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*2
shape):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

serving_default_Feature_MatrixPlaceholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ(
№

StatefulPartitionedCallStatefulPartitionedCall serving_default_Adjacency_Matrixserving_default_Feature_Matrix&DMGAttention_0/DMGAttention_0_0_weight5DMGAttention_0/DMGAttention_0_0_self_attention_weight9DMGAttention_0/DMGAttention_0_0_neighbor_attention_weight$DMGAttention_0/DMGAttention_0_0_bias&DMGAttention_0/DMGAttention_0_1_weight5DMGAttention_0/DMGAttention_0_1_self_attention_weight9DMGAttention_0/DMGAttention_0_1_neighbor_attention_weight$DMGAttention_0/DMGAttention_0_1_bias&DMGAttention_1/DMGAttention_1_0_weight5DMGAttention_1/DMGAttention_1_0_self_attention_weight9DMGAttention_1/DMGAttention_1_0_neighbor_attention_weight$DMGAttention_1/DMGAttention_1_0_bias&DMGAttention_1/DMGAttention_1_1_weight5DMGAttention_1/DMGAttention_1_1_self_attention_weight9DMGAttention_1/DMGAttention_1_1_neighbor_attention_weight$DMGAttention_1/DMGAttention_1_1_bias(DMDense_Hidden_0/DMDense_Hidden_0_weight&DMDense_Hidden_0/DMDense_Hidden_0_bias(DMDense_Hidden_1/DMDense_Hidden_1_weight&DMDense_Hidden_1/DMDense_Hidden_1_biasDMDense_OUT/DMDense_OUT_weightDMDense_OUT/DMDense_OUT_bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8 */
f*R(
&__inference_signature_wrapper_51634432
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:DMGAttention_0/DMGAttention_0_0_weight/Read/ReadVariableOpIDMGAttention_0/DMGAttention_0_0_self_attention_weight/Read/ReadVariableOpMDMGAttention_0/DMGAttention_0_0_neighbor_attention_weight/Read/ReadVariableOp8DMGAttention_0/DMGAttention_0_0_bias/Read/ReadVariableOp:DMGAttention_0/DMGAttention_0_1_weight/Read/ReadVariableOpIDMGAttention_0/DMGAttention_0_1_self_attention_weight/Read/ReadVariableOpMDMGAttention_0/DMGAttention_0_1_neighbor_attention_weight/Read/ReadVariableOp8DMGAttention_0/DMGAttention_0_1_bias/Read/ReadVariableOp:DMGAttention_1/DMGAttention_1_0_weight/Read/ReadVariableOpIDMGAttention_1/DMGAttention_1_0_self_attention_weight/Read/ReadVariableOpMDMGAttention_1/DMGAttention_1_0_neighbor_attention_weight/Read/ReadVariableOp8DMGAttention_1/DMGAttention_1_0_bias/Read/ReadVariableOp:DMGAttention_1/DMGAttention_1_1_weight/Read/ReadVariableOpIDMGAttention_1/DMGAttention_1_1_self_attention_weight/Read/ReadVariableOpMDMGAttention_1/DMGAttention_1_1_neighbor_attention_weight/Read/ReadVariableOp8DMGAttention_1/DMGAttention_1_1_bias/Read/ReadVariableOp<DMDense_Hidden_0/DMDense_Hidden_0_weight/Read/ReadVariableOp:DMDense_Hidden_0/DMDense_Hidden_0_bias/Read/ReadVariableOp<DMDense_Hidden_1/DMDense_Hidden_1_weight/Read/ReadVariableOp:DMDense_Hidden_1/DMDense_Hidden_1_bias/Read/ReadVariableOp2DMDense_OUT/DMDense_OUT_weight/Read/ReadVariableOp0DMDense_OUT/DMDense_OUT_bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8 **
f%R#
!__inference__traced_save_51636560
И

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&DMGAttention_0/DMGAttention_0_0_weight5DMGAttention_0/DMGAttention_0_0_self_attention_weight9DMGAttention_0/DMGAttention_0_0_neighbor_attention_weight$DMGAttention_0/DMGAttention_0_0_bias&DMGAttention_0/DMGAttention_0_1_weight5DMGAttention_0/DMGAttention_0_1_self_attention_weight9DMGAttention_0/DMGAttention_0_1_neighbor_attention_weight$DMGAttention_0/DMGAttention_0_1_bias&DMGAttention_1/DMGAttention_1_0_weight5DMGAttention_1/DMGAttention_1_0_self_attention_weight9DMGAttention_1/DMGAttention_1_0_neighbor_attention_weight$DMGAttention_1/DMGAttention_1_0_bias&DMGAttention_1/DMGAttention_1_1_weight5DMGAttention_1/DMGAttention_1_1_self_attention_weight9DMGAttention_1/DMGAttention_1_1_neighbor_attention_weight$DMGAttention_1/DMGAttention_1_1_bias(DMDense_Hidden_0/DMDense_Hidden_0_weight&DMDense_Hidden_0/DMDense_Hidden_0_bias(DMDense_Hidden_1/DMDense_Hidden_1_weight&DMDense_Hidden_1/DMDense_Hidden_1_biasDMDense_OUT/DMDense_OUT_weightDMDense_OUT/DMDense_OUT_biastotal_1count_1totalcount*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8 *-
f(R&
$__inference__traced_restore_51636648дц
Н+
ж	
C__inference_model_layer_call_and_return_conditional_losses_51634169

inputs
inputs_1)
dmgattention_0_51634116:( )
dmgattention_0_51634118: )
dmgattention_0_51634120: %
dmgattention_0_51634122: )
dmgattention_0_51634124:( )
dmgattention_0_51634126: )
dmgattention_0_51634128: %
dmgattention_0_51634130: )
dmgattention_1_51634134:  )
dmgattention_1_51634136: )
dmgattention_1_51634138: %
dmgattention_1_51634140: )
dmgattention_1_51634142:  )
dmgattention_1_51634144: )
dmgattention_1_51634146: %
dmgattention_1_51634148: +
dmdense_hidden_0_51634153: '
dmdense_hidden_0_51634155:+
dmdense_hidden_1_51634158:'
dmdense_hidden_1_51634160:&
dmdense_out_51634163:"
dmdense_out_51634165:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂ&DMGAttention_0/StatefulPartitionedCallЂ&DMGAttention_1/StatefulPartitionedCallі
&DMGAttention_0/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dmgattention_0_51634116dmgattention_0_51634118dmgattention_0_51634120dmgattention_0_51634122dmgattention_0_51634124dmgattention_0_51634126dmgattention_0_51634128dmgattention_0_51634130*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51634038Ц
&DMGAttention_1/StatefulPartitionedCallStatefulPartitionedCall/DMGAttention_0/StatefulPartitionedCall:output:0/DMGAttention_0/StatefulPartitionedCall:output:1dmgattention_1_51634134dmgattention_1_51634136dmgattention_1_51634138dmgattention_1_51634140dmgattention_1_51634142dmgattention_1_51634144dmgattention_1_51634146dmgattention_1_51634148*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633757
DMGReduce_1/PartitionedCallPartitionedCall/DMGAttention_1/StatefulPartitionedCall:output:0/DMGAttention_1/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51633360И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51634153dmdense_hidden_0_51634155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51633373Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51634158dmdense_hidden_1_51634160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51633390Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51634163dmdense_out_51634165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51633406{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall'^DMGAttention_0/StatefulPartitionedCall'^DMGAttention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2P
&DMGAttention_0/StatefulPartitionedCall&DMGAttention_0/StatefulPartitionedCall2P
&DMGAttention_1/StatefulPartitionedCall&DMGAttention_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о	

I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51636458
input_tensor0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMulMatMulinput_tensorMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:U Q
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
Д


N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51636439
input_tensor0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMulMatMulinput_tensorMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:U Q
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
ЭЫ

#__inference__wrapped_model_51632906
feature_matrix
adjacency_matrixF
4model_dmgattention_0_shape_1_readvariableop_resource:( F
4model_dmgattention_0_shape_3_readvariableop_resource: F
4model_dmgattention_0_shape_5_readvariableop_resource: B
4model_dmgattention_0_biasadd_readvariableop_resource: F
4model_dmgattention_0_shape_9_readvariableop_resource:( G
5model_dmgattention_0_shape_11_readvariableop_resource: G
5model_dmgattention_0_shape_13_readvariableop_resource: D
6model_dmgattention_0_biasadd_1_readvariableop_resource: F
4model_dmgattention_1_shape_1_readvariableop_resource:  F
4model_dmgattention_1_shape_3_readvariableop_resource: F
4model_dmgattention_1_shape_5_readvariableop_resource: B
4model_dmgattention_1_biasadd_readvariableop_resource: F
4model_dmgattention_1_shape_9_readvariableop_resource:  G
5model_dmgattention_1_shape_11_readvariableop_resource: G
5model_dmgattention_1_shape_13_readvariableop_resource: D
6model_dmgattention_1_biasadd_1_readvariableop_resource: G
5model_dmdense_hidden_0_matmul_readvariableop_resource: D
6model_dmdense_hidden_0_biasadd_readvariableop_resource:G
5model_dmdense_hidden_1_matmul_readvariableop_resource:D
6model_dmdense_hidden_1_biasadd_readvariableop_resource:B
0model_dmdense_out_matmul_readvariableop_resource:?
1model_dmdense_out_biasadd_readvariableop_resource:
identityЂ-model/DMDense_Hidden_0/BiasAdd/ReadVariableOpЂ,model/DMDense_Hidden_0/MatMul/ReadVariableOpЂ-model/DMDense_Hidden_1/BiasAdd/ReadVariableOpЂ,model/DMDense_Hidden_1/MatMul/ReadVariableOpЂ(model/DMDense_OUT/BiasAdd/ReadVariableOpЂ'model/DMDense_OUT/MatMul/ReadVariableOpЂ+model/DMGAttention_0/BiasAdd/ReadVariableOpЂ-model/DMGAttention_0/BiasAdd_1/ReadVariableOpЂ-model/DMGAttention_0/transpose/ReadVariableOpЂ/model/DMGAttention_0/transpose_1/ReadVariableOpЂ/model/DMGAttention_0/transpose_2/ReadVariableOpЂ/model/DMGAttention_0/transpose_5/ReadVariableOpЂ/model/DMGAttention_0/transpose_6/ReadVariableOpЂ/model/DMGAttention_0/transpose_7/ReadVariableOpЂ+model/DMGAttention_1/BiasAdd/ReadVariableOpЂ-model/DMGAttention_1/BiasAdd_1/ReadVariableOpЂ-model/DMGAttention_1/transpose/ReadVariableOpЂ/model/DMGAttention_1/transpose_1/ReadVariableOpЂ/model/DMGAttention_1/transpose_2/ReadVariableOpЂ/model/DMGAttention_1/transpose_5/ReadVariableOpЂ/model/DMGAttention_1/transpose_6/ReadVariableOpЂ/model/DMGAttention_1/transpose_7/ReadVariableOpX
model/DMGAttention_0/ShapeShapefeature_matrix*
T0*
_output_shapes
:{
model/DMGAttention_0/unstackUnpack#model/DMGAttention_0/Shape:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_0/Shape_1/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_1_readvariableop_resource*
_output_shapes

:( *
dtype0m
model/DMGAttention_0/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"(       }
model/DMGAttention_0/unstack_1Unpack%model/DMGAttention_0/Shape_1:output:0*
T0*
_output_shapes
: : *	
nums
"model/DMGAttention_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   
model/DMGAttention_0/ReshapeReshapefeature_matrix+model/DMGAttention_0/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(Ђ
-model/DMGAttention_0/transpose/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_1_readvariableop_resource*
_output_shapes

:( *
dtype0t
#model/DMGAttention_0/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
model/DMGAttention_0/transpose	Transpose5model/DMGAttention_0/transpose/ReadVariableOp:value:0,model/DMGAttention_0/transpose/perm:output:0*
T0*
_output_shapes

:( u
$model/DMGAttention_0/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџЅ
model/DMGAttention_0/Reshape_1Reshape"model/DMGAttention_0/transpose:y:0-model/DMGAttention_0/Reshape_1/shape:output:0*
T0*
_output_shapes

:( Ї
model/DMGAttention_0/MatMulMatMul%model/DMGAttention_0/Reshape:output:0'model/DMGAttention_0/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ h
&model/DMGAttention_0/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : й
$model/DMGAttention_0/Reshape_2/shapePack%model/DMGAttention_0/unstack:output:0%model/DMGAttention_0/unstack:output:1/model/DMGAttention_0/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:О
model/DMGAttention_0/Reshape_2Reshape%model/DMGAttention_0/MatMul:product:0-model/DMGAttention_0/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ s
model/DMGAttention_0/Shape_2Shape'model/DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_2Unpack%model/DMGAttention_0/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_0/Shape_3/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0m
model/DMGAttention_0/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       }
model/DMGAttention_0/unstack_3Unpack%model/DMGAttention_0/Shape_3:output:0*
T0*
_output_shapes
: : *	
numu
$model/DMGAttention_0/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Г
model/DMGAttention_0/Reshape_3Reshape'model/DMGAttention_0/Reshape_2:output:0-model/DMGAttention_0/Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Є
/model/DMGAttention_0/transpose_1/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_0/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_0/transpose_1	Transpose7model/DMGAttention_0/transpose_1/ReadVariableOp:value:0.model/DMGAttention_0/transpose_1/perm:output:0*
T0*
_output_shapes

: u
$model/DMGAttention_0/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЇ
model/DMGAttention_0/Reshape_4Reshape$model/DMGAttention_0/transpose_1:y:0-model/DMGAttention_0/Reshape_4/shape:output:0*
T0*
_output_shapes

: Ћ
model/DMGAttention_0/MatMul_1MatMul'model/DMGAttention_0/Reshape_3:output:0'model/DMGAttention_0/Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
&model/DMGAttention_0/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :н
$model/DMGAttention_0/Reshape_5/shapePack'model/DMGAttention_0/unstack_2:output:0'model/DMGAttention_0/unstack_2:output:1/model/DMGAttention_0/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Р
model/DMGAttention_0/Reshape_5Reshape'model/DMGAttention_0/MatMul_1:product:0-model/DMGAttention_0/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџs
model/DMGAttention_0/Shape_4Shape'model/DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_4Unpack%model/DMGAttention_0/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_0/Shape_5/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0m
model/DMGAttention_0/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       }
model/DMGAttention_0/unstack_5Unpack%model/DMGAttention_0/Shape_5:output:0*
T0*
_output_shapes
: : *	
numu
$model/DMGAttention_0/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Г
model/DMGAttention_0/Reshape_6Reshape'model/DMGAttention_0/Reshape_2:output:0-model/DMGAttention_0/Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Є
/model/DMGAttention_0/transpose_2/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_0/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_0/transpose_2	Transpose7model/DMGAttention_0/transpose_2/ReadVariableOp:value:0.model/DMGAttention_0/transpose_2/perm:output:0*
T0*
_output_shapes

: u
$model/DMGAttention_0/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЇ
model/DMGAttention_0/Reshape_7Reshape$model/DMGAttention_0/transpose_2:y:0-model/DMGAttention_0/Reshape_7/shape:output:0*
T0*
_output_shapes

: Ћ
model/DMGAttention_0/MatMul_2MatMul'model/DMGAttention_0/Reshape_6:output:0'model/DMGAttention_0/Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
&model/DMGAttention_0/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :н
$model/DMGAttention_0/Reshape_8/shapePack'model/DMGAttention_0/unstack_4:output:0'model/DMGAttention_0/unstack_4:output:1/model/DMGAttention_0/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:Р
model/DMGAttention_0/Reshape_8Reshape'model/DMGAttention_0/MatMul_2:product:0-model/DMGAttention_0/Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_0/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Х
 model/DMGAttention_0/transpose_3	Transpose'model/DMGAttention_0/Reshape_8:output:0.model/DMGAttention_0/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџИ
model/DMGAttention_0/addAddV2'model/DMGAttention_0/Reshape_5:output:0$model/DMGAttention_0/transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGAttention_0/LeakyRelu	LeakyRelumodel/DMGAttention_0/add:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ_
model/DMGAttention_0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/DMGAttention_0/subSub#model/DMGAttention_0/sub/x:output:0adjacency_matrix*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ_
model/DMGAttention_0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаЊ
model/DMGAttention_0/mulMul#model/DMGAttention_0/mul/x:output:0model/DMGAttention_0/sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЗ
model/DMGAttention_0/add_1AddV2,model/DMGAttention_0/LeakyRelu:activations:0model/DMGAttention_0/mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGAttention_0/SoftmaxSoftmaxmodel/DMGAttention_0/add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџr
model/DMGAttention_0/Shape_6Shape&model/DMGAttention_0/Softmax:softmax:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_6Unpack%model/DMGAttention_0/Shape_6:output:0*
T0*
_output_shapes
: : : *	
nums
model/DMGAttention_0/Shape_7Shape'model/DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_7Unpack%model/DMGAttention_0/Shape_7:output:0*
T0*
_output_shapes
: : : *	
numq
&model/DMGAttention_0/Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
$model/DMGAttention_0/Reshape_9/shapePack/model/DMGAttention_0/Reshape_9/shape/0:output:0'model/DMGAttention_0/unstack_6:output:2*
N*
T0*
_output_shapes
:Л
model/DMGAttention_0/Reshape_9Reshape&model/DMGAttention_0/Softmax:softmax:0-model/DMGAttention_0/Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_0/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Х
 model/DMGAttention_0/transpose_4	Transpose'model/DMGAttention_0/Reshape_2:output:0.model/DMGAttention_0/transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
'model/DMGAttention_0/Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЖ
%model/DMGAttention_0/Reshape_10/shapePack'model/DMGAttention_0/unstack_7:output:10model/DMGAttention_0/Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:Л
model/DMGAttention_0/Reshape_10Reshape$model/DMGAttention_0/transpose_4:y:0.model/DMGAttention_0/Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЕ
model/DMGAttention_0/MatMul_3MatMul'model/DMGAttention_0/Reshape_9:output:0(model/DMGAttention_0/Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџi
'model/DMGAttention_0/Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
%model/DMGAttention_0/Reshape_11/shapePack'model/DMGAttention_0/unstack_6:output:0'model/DMGAttention_0/unstack_6:output:1'model/DMGAttention_0/unstack_7:output:00model/DMGAttention_0/Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:Я
model/DMGAttention_0/Reshape_11Reshape'model/DMGAttention_0/MatMul_3:product:0.model/DMGAttention_0/Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
+model/DMGAttention_0/BiasAdd/ReadVariableOpReadVariableOp4model_dmgattention_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0в
model/DMGAttention_0/BiasAddBiasAdd(model/DMGAttention_0/Reshape_11:output:03model/DMGAttention_0/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Z
model/DMGAttention_0/Shape_8Shapefeature_matrix*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_8Unpack%model/DMGAttention_0/Shape_8:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_0/Shape_9/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_9_readvariableop_resource*
_output_shapes

:( *
dtype0m
model/DMGAttention_0/Shape_9Const*
_output_shapes
:*
dtype0*
valueB"(       }
model/DMGAttention_0/unstack_9Unpack%model/DMGAttention_0/Shape_9:output:0*
T0*
_output_shapes
: : *	
numv
%model/DMGAttention_0/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   
model/DMGAttention_0/Reshape_12Reshapefeature_matrix.model/DMGAttention_0/Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(Є
/model/DMGAttention_0/transpose_5/ReadVariableOpReadVariableOp4model_dmgattention_0_shape_9_readvariableop_resource*
_output_shapes

:( *
dtype0v
%model/DMGAttention_0/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_0/transpose_5	Transpose7model/DMGAttention_0/transpose_5/ReadVariableOp:value:0.model/DMGAttention_0/transpose_5/perm:output:0*
T0*
_output_shapes

:( v
%model/DMGAttention_0/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџЉ
model/DMGAttention_0/Reshape_13Reshape$model/DMGAttention_0/transpose_5:y:0.model/DMGAttention_0/Reshape_13/shape:output:0*
T0*
_output_shapes

:( ­
model/DMGAttention_0/MatMul_4MatMul(model/DMGAttention_0/Reshape_12:output:0(model/DMGAttention_0/Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ i
'model/DMGAttention_0/Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : п
%model/DMGAttention_0/Reshape_14/shapePack'model/DMGAttention_0/unstack_8:output:0'model/DMGAttention_0/unstack_8:output:10model/DMGAttention_0/Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:Т
model/DMGAttention_0/Reshape_14Reshape'model/DMGAttention_0/MatMul_4:product:0.model/DMGAttention_0/Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ u
model/DMGAttention_0/Shape_10Shape(model/DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_10Unpack&model/DMGAttention_0/Shape_10:output:0*
T0*
_output_shapes
: : : *	
numЂ
,model/DMGAttention_0/Shape_11/ReadVariableOpReadVariableOp5model_dmgattention_0_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0n
model/DMGAttention_0/Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       
model/DMGAttention_0/unstack_11Unpack&model/DMGAttention_0/Shape_11:output:0*
T0*
_output_shapes
: : *	
numv
%model/DMGAttention_0/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ж
model/DMGAttention_0/Reshape_15Reshape(model/DMGAttention_0/Reshape_14:output:0.model/DMGAttention_0/Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ѕ
/model/DMGAttention_0/transpose_6/ReadVariableOpReadVariableOp5model_dmgattention_0_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_0/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_0/transpose_6	Transpose7model/DMGAttention_0/transpose_6/ReadVariableOp:value:0.model/DMGAttention_0/transpose_6/perm:output:0*
T0*
_output_shapes

: v
%model/DMGAttention_0/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЉ
model/DMGAttention_0/Reshape_16Reshape$model/DMGAttention_0/transpose_6:y:0.model/DMGAttention_0/Reshape_16/shape:output:0*
T0*
_output_shapes

: ­
model/DMGAttention_0/MatMul_5MatMul(model/DMGAttention_0/Reshape_15:output:0(model/DMGAttention_0/Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
'model/DMGAttention_0/Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :с
%model/DMGAttention_0/Reshape_17/shapePack(model/DMGAttention_0/unstack_10:output:0(model/DMGAttention_0/unstack_10:output:10model/DMGAttention_0/Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:Т
model/DMGAttention_0/Reshape_17Reshape'model/DMGAttention_0/MatMul_5:product:0.model/DMGAttention_0/Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџu
model/DMGAttention_0/Shape_12Shape(model/DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_12Unpack&model/DMGAttention_0/Shape_12:output:0*
T0*
_output_shapes
: : : *	
numЂ
,model/DMGAttention_0/Shape_13/ReadVariableOpReadVariableOp5model_dmgattention_0_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0n
model/DMGAttention_0/Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       
model/DMGAttention_0/unstack_13Unpack&model/DMGAttention_0/Shape_13:output:0*
T0*
_output_shapes
: : *	
numv
%model/DMGAttention_0/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ж
model/DMGAttention_0/Reshape_18Reshape(model/DMGAttention_0/Reshape_14:output:0.model/DMGAttention_0/Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ѕ
/model/DMGAttention_0/transpose_7/ReadVariableOpReadVariableOp5model_dmgattention_0_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_0/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_0/transpose_7	Transpose7model/DMGAttention_0/transpose_7/ReadVariableOp:value:0.model/DMGAttention_0/transpose_7/perm:output:0*
T0*
_output_shapes

: v
%model/DMGAttention_0/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЉ
model/DMGAttention_0/Reshape_19Reshape$model/DMGAttention_0/transpose_7:y:0.model/DMGAttention_0/Reshape_19/shape:output:0*
T0*
_output_shapes

: ­
model/DMGAttention_0/MatMul_6MatMul(model/DMGAttention_0/Reshape_18:output:0(model/DMGAttention_0/Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
'model/DMGAttention_0/Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :с
%model/DMGAttention_0/Reshape_20/shapePack(model/DMGAttention_0/unstack_12:output:0(model/DMGAttention_0/unstack_12:output:10model/DMGAttention_0/Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:Т
model/DMGAttention_0/Reshape_20Reshape'model/DMGAttention_0/MatMul_6:product:0.model/DMGAttention_0/Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_0/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
 model/DMGAttention_0/transpose_8	Transpose(model/DMGAttention_0/Reshape_20:output:0.model/DMGAttention_0/transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЛ
model/DMGAttention_0/add_2AddV2(model/DMGAttention_0/Reshape_17:output:0$model/DMGAttention_0/transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 model/DMGAttention_0/LeakyRelu_1	LeakyRelumodel/DMGAttention_0/add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџa
model/DMGAttention_0/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
model/DMGAttention_0/sub_1Sub%model/DMGAttention_0/sub_1/x:output:0adjacency_matrix*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџa
model/DMGAttention_0/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаА
model/DMGAttention_0/mul_1Mul%model/DMGAttention_0/mul_1/x:output:0model/DMGAttention_0/sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
model/DMGAttention_0/add_3AddV2.model/DMGAttention_0/LeakyRelu_1:activations:0model/DMGAttention_0/mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGAttention_0/Softmax_1Softmaxmodel/DMGAttention_0/add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
model/DMGAttention_0/Shape_14Shape(model/DMGAttention_0/Softmax_1:softmax:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_14Unpack&model/DMGAttention_0/Shape_14:output:0*
T0*
_output_shapes
: : : *	
numu
model/DMGAttention_0/Shape_15Shape(model/DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:
model/DMGAttention_0/unstack_15Unpack&model/DMGAttention_0/Shape_15:output:0*
T0*
_output_shapes
: : : *	
numr
'model/DMGAttention_0/Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЗ
%model/DMGAttention_0/Reshape_21/shapePack0model/DMGAttention_0/Reshape_21/shape/0:output:0(model/DMGAttention_0/unstack_14:output:2*
N*
T0*
_output_shapes
:П
model/DMGAttention_0/Reshape_21Reshape(model/DMGAttention_0/Softmax_1:softmax:0.model/DMGAttention_0/Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_0/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
 model/DMGAttention_0/transpose_9	Transpose(model/DMGAttention_0/Reshape_14:output:0.model/DMGAttention_0/transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
'model/DMGAttention_0/Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЗ
%model/DMGAttention_0/Reshape_22/shapePack(model/DMGAttention_0/unstack_15:output:10model/DMGAttention_0/Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:Л
model/DMGAttention_0/Reshape_22Reshape$model/DMGAttention_0/transpose_9:y:0.model/DMGAttention_0/Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЖ
model/DMGAttention_0/MatMul_7MatMul(model/DMGAttention_0/Reshape_21:output:0(model/DMGAttention_0/Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџi
'model/DMGAttention_0/Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
%model/DMGAttention_0/Reshape_23/shapePack(model/DMGAttention_0/unstack_14:output:0(model/DMGAttention_0/unstack_14:output:1(model/DMGAttention_0/unstack_15:output:00model/DMGAttention_0/Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:Я
model/DMGAttention_0/Reshape_23Reshape'model/DMGAttention_0/MatMul_7:product:0.model/DMGAttention_0/Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ  
-model/DMGAttention_0/BiasAdd_1/ReadVariableOpReadVariableOp6model_dmgattention_0_biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0ж
model/DMGAttention_0/BiasAdd_1BiasAdd(model/DMGAttention_0/Reshape_23:output:05model/DMGAttention_0/BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ы
model/DMGAttention_0/stackPack%model/DMGAttention_0/BiasAdd:output:0'model/DMGAttention_0/BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ m
+model/DMGAttention_0/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : Ш
model/DMGAttention_0/MeanMean#model/DMGAttention_0/stack:output:04model/DMGAttention_0/Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ }
(model/DMGAttention_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*model/DMGAttention_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*model/DMGAttention_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         љ
"model/DMGAttention_0/strided_sliceStridedSlice"model/DMGAttention_0/Mean:output:01model/DMGAttention_0/strided_slice/stack:output:03model/DMGAttention_0/strided_slice/stack_1:output:03model/DMGAttention_0/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_mask
model/DMGAttention_0/EluElu+model/DMGAttention_0/strided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ p
model/DMGAttention_1/ShapeShape&model/DMGAttention_0/Elu:activations:0*
T0*
_output_shapes
:{
model/DMGAttention_1/unstackUnpack#model/DMGAttention_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_1/Shape_1/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_1_readvariableop_resource*
_output_shapes

:  *
dtype0m
model/DMGAttention_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"        }
model/DMGAttention_1/unstack_1Unpack%model/DMGAttention_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
nums
"model/DMGAttention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ў
model/DMGAttention_1/ReshapeReshape&model/DMGAttention_0/Elu:activations:0+model/DMGAttention_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
-model/DMGAttention_1/transpose/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_1_readvariableop_resource*
_output_shapes

:  *
dtype0t
#model/DMGAttention_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Й
model/DMGAttention_1/transpose	Transpose5model/DMGAttention_1/transpose/ReadVariableOp:value:0,model/DMGAttention_1/transpose/perm:output:0*
T0*
_output_shapes

:  u
$model/DMGAttention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЅ
model/DMGAttention_1/Reshape_1Reshape"model/DMGAttention_1/transpose:y:0-model/DMGAttention_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:  Ї
model/DMGAttention_1/MatMulMatMul%model/DMGAttention_1/Reshape:output:0'model/DMGAttention_1/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ h
&model/DMGAttention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : й
$model/DMGAttention_1/Reshape_2/shapePack%model/DMGAttention_1/unstack:output:0%model/DMGAttention_1/unstack:output:1/model/DMGAttention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:О
model/DMGAttention_1/Reshape_2Reshape%model/DMGAttention_1/MatMul:product:0-model/DMGAttention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ s
model/DMGAttention_1/Shape_2Shape'model/DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_2Unpack%model/DMGAttention_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_1/Shape_3/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0m
model/DMGAttention_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       }
model/DMGAttention_1/unstack_3Unpack%model/DMGAttention_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
numu
$model/DMGAttention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Г
model/DMGAttention_1/Reshape_3Reshape'model/DMGAttention_1/Reshape_2:output:0-model/DMGAttention_1/Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Є
/model/DMGAttention_1/transpose_1/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_1/transpose_1	Transpose7model/DMGAttention_1/transpose_1/ReadVariableOp:value:0.model/DMGAttention_1/transpose_1/perm:output:0*
T0*
_output_shapes

: u
$model/DMGAttention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЇ
model/DMGAttention_1/Reshape_4Reshape$model/DMGAttention_1/transpose_1:y:0-model/DMGAttention_1/Reshape_4/shape:output:0*
T0*
_output_shapes

: Ћ
model/DMGAttention_1/MatMul_1MatMul'model/DMGAttention_1/Reshape_3:output:0'model/DMGAttention_1/Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
&model/DMGAttention_1/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :н
$model/DMGAttention_1/Reshape_5/shapePack'model/DMGAttention_1/unstack_2:output:0'model/DMGAttention_1/unstack_2:output:1/model/DMGAttention_1/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Р
model/DMGAttention_1/Reshape_5Reshape'model/DMGAttention_1/MatMul_1:product:0-model/DMGAttention_1/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџs
model/DMGAttention_1/Shape_4Shape'model/DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_4Unpack%model/DMGAttention_1/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_1/Shape_5/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0m
model/DMGAttention_1/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       }
model/DMGAttention_1/unstack_5Unpack%model/DMGAttention_1/Shape_5:output:0*
T0*
_output_shapes
: : *	
numu
$model/DMGAttention_1/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Г
model/DMGAttention_1/Reshape_6Reshape'model/DMGAttention_1/Reshape_2:output:0-model/DMGAttention_1/Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Є
/model/DMGAttention_1/transpose_2/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_1/transpose_2	Transpose7model/DMGAttention_1/transpose_2/ReadVariableOp:value:0.model/DMGAttention_1/transpose_2/perm:output:0*
T0*
_output_shapes

: u
$model/DMGAttention_1/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЇ
model/DMGAttention_1/Reshape_7Reshape$model/DMGAttention_1/transpose_2:y:0-model/DMGAttention_1/Reshape_7/shape:output:0*
T0*
_output_shapes

: Ћ
model/DMGAttention_1/MatMul_2MatMul'model/DMGAttention_1/Reshape_6:output:0'model/DMGAttention_1/Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
&model/DMGAttention_1/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :н
$model/DMGAttention_1/Reshape_8/shapePack'model/DMGAttention_1/unstack_4:output:0'model/DMGAttention_1/unstack_4:output:1/model/DMGAttention_1/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:Р
model/DMGAttention_1/Reshape_8Reshape'model/DMGAttention_1/MatMul_2:product:0-model/DMGAttention_1/Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Х
 model/DMGAttention_1/transpose_3	Transpose'model/DMGAttention_1/Reshape_8:output:0.model/DMGAttention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџИ
model/DMGAttention_1/addAddV2'model/DMGAttention_1/Reshape_5:output:0$model/DMGAttention_1/transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGAttention_1/LeakyRelu	LeakyRelumodel/DMGAttention_1/add:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ_
model/DMGAttention_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/DMGAttention_1/subSub#model/DMGAttention_1/sub/x:output:0adjacency_matrix*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ_
model/DMGAttention_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаЊ
model/DMGAttention_1/mulMul#model/DMGAttention_1/mul/x:output:0model/DMGAttention_1/sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЗ
model/DMGAttention_1/add_1AddV2,model/DMGAttention_1/LeakyRelu:activations:0model/DMGAttention_1/mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGAttention_1/SoftmaxSoftmaxmodel/DMGAttention_1/add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџr
model/DMGAttention_1/Shape_6Shape&model/DMGAttention_1/Softmax:softmax:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_6Unpack%model/DMGAttention_1/Shape_6:output:0*
T0*
_output_shapes
: : : *	
nums
model/DMGAttention_1/Shape_7Shape'model/DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_7Unpack%model/DMGAttention_1/Shape_7:output:0*
T0*
_output_shapes
: : : *	
numq
&model/DMGAttention_1/Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
$model/DMGAttention_1/Reshape_9/shapePack/model/DMGAttention_1/Reshape_9/shape/0:output:0'model/DMGAttention_1/unstack_6:output:2*
N*
T0*
_output_shapes
:Л
model/DMGAttention_1/Reshape_9Reshape&model/DMGAttention_1/Softmax:softmax:0-model/DMGAttention_1/Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_1/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Х
 model/DMGAttention_1/transpose_4	Transpose'model/DMGAttention_1/Reshape_2:output:0.model/DMGAttention_1/transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
'model/DMGAttention_1/Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЖ
%model/DMGAttention_1/Reshape_10/shapePack'model/DMGAttention_1/unstack_7:output:10model/DMGAttention_1/Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:Л
model/DMGAttention_1/Reshape_10Reshape$model/DMGAttention_1/transpose_4:y:0.model/DMGAttention_1/Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЕ
model/DMGAttention_1/MatMul_3MatMul'model/DMGAttention_1/Reshape_9:output:0(model/DMGAttention_1/Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџi
'model/DMGAttention_1/Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
%model/DMGAttention_1/Reshape_11/shapePack'model/DMGAttention_1/unstack_6:output:0'model/DMGAttention_1/unstack_6:output:1'model/DMGAttention_1/unstack_7:output:00model/DMGAttention_1/Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:Я
model/DMGAttention_1/Reshape_11Reshape'model/DMGAttention_1/MatMul_3:product:0.model/DMGAttention_1/Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
+model/DMGAttention_1/BiasAdd/ReadVariableOpReadVariableOp4model_dmgattention_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0в
model/DMGAttention_1/BiasAddBiasAdd(model/DMGAttention_1/Reshape_11:output:03model/DMGAttention_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
model/DMGAttention_1/Shape_8Shape&model/DMGAttention_0/Elu:activations:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_8Unpack%model/DMGAttention_1/Shape_8:output:0*
T0*
_output_shapes
: : : *	
num 
+model/DMGAttention_1/Shape_9/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_9_readvariableop_resource*
_output_shapes

:  *
dtype0m
model/DMGAttention_1/Shape_9Const*
_output_shapes
:*
dtype0*
valueB"        }
model/DMGAttention_1/unstack_9Unpack%model/DMGAttention_1/Shape_9:output:0*
T0*
_output_shapes
: : *	
numv
%model/DMGAttention_1/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Д
model/DMGAttention_1/Reshape_12Reshape&model/DMGAttention_0/Elu:activations:0.model/DMGAttention_1/Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Є
/model/DMGAttention_1/transpose_5/ReadVariableOpReadVariableOp4model_dmgattention_1_shape_9_readvariableop_resource*
_output_shapes

:  *
dtype0v
%model/DMGAttention_1/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_1/transpose_5	Transpose7model/DMGAttention_1/transpose_5/ReadVariableOp:value:0.model/DMGAttention_1/transpose_5/perm:output:0*
T0*
_output_shapes

:  v
%model/DMGAttention_1/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЉ
model/DMGAttention_1/Reshape_13Reshape$model/DMGAttention_1/transpose_5:y:0.model/DMGAttention_1/Reshape_13/shape:output:0*
T0*
_output_shapes

:  ­
model/DMGAttention_1/MatMul_4MatMul(model/DMGAttention_1/Reshape_12:output:0(model/DMGAttention_1/Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ i
'model/DMGAttention_1/Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : п
%model/DMGAttention_1/Reshape_14/shapePack'model/DMGAttention_1/unstack_8:output:0'model/DMGAttention_1/unstack_8:output:10model/DMGAttention_1/Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:Т
model/DMGAttention_1/Reshape_14Reshape'model/DMGAttention_1/MatMul_4:product:0.model/DMGAttention_1/Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ u
model/DMGAttention_1/Shape_10Shape(model/DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_10Unpack&model/DMGAttention_1/Shape_10:output:0*
T0*
_output_shapes
: : : *	
numЂ
,model/DMGAttention_1/Shape_11/ReadVariableOpReadVariableOp5model_dmgattention_1_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0n
model/DMGAttention_1/Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       
model/DMGAttention_1/unstack_11Unpack&model/DMGAttention_1/Shape_11:output:0*
T0*
_output_shapes
: : *	
numv
%model/DMGAttention_1/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ж
model/DMGAttention_1/Reshape_15Reshape(model/DMGAttention_1/Reshape_14:output:0.model/DMGAttention_1/Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ѕ
/model/DMGAttention_1/transpose_6/ReadVariableOpReadVariableOp5model_dmgattention_1_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_1/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_1/transpose_6	Transpose7model/DMGAttention_1/transpose_6/ReadVariableOp:value:0.model/DMGAttention_1/transpose_6/perm:output:0*
T0*
_output_shapes

: v
%model/DMGAttention_1/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЉ
model/DMGAttention_1/Reshape_16Reshape$model/DMGAttention_1/transpose_6:y:0.model/DMGAttention_1/Reshape_16/shape:output:0*
T0*
_output_shapes

: ­
model/DMGAttention_1/MatMul_5MatMul(model/DMGAttention_1/Reshape_15:output:0(model/DMGAttention_1/Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
'model/DMGAttention_1/Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :с
%model/DMGAttention_1/Reshape_17/shapePack(model/DMGAttention_1/unstack_10:output:0(model/DMGAttention_1/unstack_10:output:10model/DMGAttention_1/Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:Т
model/DMGAttention_1/Reshape_17Reshape'model/DMGAttention_1/MatMul_5:product:0.model/DMGAttention_1/Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџu
model/DMGAttention_1/Shape_12Shape(model/DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_12Unpack&model/DMGAttention_1/Shape_12:output:0*
T0*
_output_shapes
: : : *	
numЂ
,model/DMGAttention_1/Shape_13/ReadVariableOpReadVariableOp5model_dmgattention_1_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0n
model/DMGAttention_1/Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       
model/DMGAttention_1/unstack_13Unpack&model/DMGAttention_1/Shape_13:output:0*
T0*
_output_shapes
: : *	
numv
%model/DMGAttention_1/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ж
model/DMGAttention_1/Reshape_18Reshape(model/DMGAttention_1/Reshape_14:output:0.model/DMGAttention_1/Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ѕ
/model/DMGAttention_1/transpose_7/ReadVariableOpReadVariableOp5model_dmgattention_1_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0v
%model/DMGAttention_1/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       П
 model/DMGAttention_1/transpose_7	Transpose7model/DMGAttention_1/transpose_7/ReadVariableOp:value:0.model/DMGAttention_1/transpose_7/perm:output:0*
T0*
_output_shapes

: v
%model/DMGAttention_1/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџЉ
model/DMGAttention_1/Reshape_19Reshape$model/DMGAttention_1/transpose_7:y:0.model/DMGAttention_1/Reshape_19/shape:output:0*
T0*
_output_shapes

: ­
model/DMGAttention_1/MatMul_6MatMul(model/DMGAttention_1/Reshape_18:output:0(model/DMGAttention_1/Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
'model/DMGAttention_1/Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :с
%model/DMGAttention_1/Reshape_20/shapePack(model/DMGAttention_1/unstack_12:output:0(model/DMGAttention_1/unstack_12:output:10model/DMGAttention_1/Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:Т
model/DMGAttention_1/Reshape_20Reshape'model/DMGAttention_1/MatMul_6:product:0.model/DMGAttention_1/Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_1/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
 model/DMGAttention_1/transpose_8	Transpose(model/DMGAttention_1/Reshape_20:output:0.model/DMGAttention_1/transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЛ
model/DMGAttention_1/add_2AddV2(model/DMGAttention_1/Reshape_17:output:0$model/DMGAttention_1/transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 model/DMGAttention_1/LeakyRelu_1	LeakyRelumodel/DMGAttention_1/add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџa
model/DMGAttention_1/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
model/DMGAttention_1/sub_1Sub%model/DMGAttention_1/sub_1/x:output:0adjacency_matrix*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџa
model/DMGAttention_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаА
model/DMGAttention_1/mul_1Mul%model/DMGAttention_1/mul_1/x:output:0model/DMGAttention_1/sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
model/DMGAttention_1/add_3AddV2.model/DMGAttention_1/LeakyRelu_1:activations:0model/DMGAttention_1/mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGAttention_1/Softmax_1Softmaxmodel/DMGAttention_1/add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
model/DMGAttention_1/Shape_14Shape(model/DMGAttention_1/Softmax_1:softmax:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_14Unpack&model/DMGAttention_1/Shape_14:output:0*
T0*
_output_shapes
: : : *	
numu
model/DMGAttention_1/Shape_15Shape(model/DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:
model/DMGAttention_1/unstack_15Unpack&model/DMGAttention_1/Shape_15:output:0*
T0*
_output_shapes
: : : *	
numr
'model/DMGAttention_1/Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЗ
%model/DMGAttention_1/Reshape_21/shapePack0model/DMGAttention_1/Reshape_21/shape/0:output:0(model/DMGAttention_1/unstack_14:output:2*
N*
T0*
_output_shapes
:П
model/DMGAttention_1/Reshape_21Reshape(model/DMGAttention_1/Softmax_1:softmax:0.model/DMGAttention_1/Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџz
%model/DMGAttention_1/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
 model/DMGAttention_1/transpose_9	Transpose(model/DMGAttention_1/Reshape_14:output:0.model/DMGAttention_1/transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
'model/DMGAttention_1/Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЗ
%model/DMGAttention_1/Reshape_22/shapePack(model/DMGAttention_1/unstack_15:output:10model/DMGAttention_1/Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:Л
model/DMGAttention_1/Reshape_22Reshape$model/DMGAttention_1/transpose_9:y:0.model/DMGAttention_1/Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЖ
model/DMGAttention_1/MatMul_7MatMul(model/DMGAttention_1/Reshape_21:output:0(model/DMGAttention_1/Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџi
'model/DMGAttention_1/Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
%model/DMGAttention_1/Reshape_23/shapePack(model/DMGAttention_1/unstack_14:output:0(model/DMGAttention_1/unstack_14:output:1(model/DMGAttention_1/unstack_15:output:00model/DMGAttention_1/Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:Я
model/DMGAttention_1/Reshape_23Reshape'model/DMGAttention_1/MatMul_7:product:0.model/DMGAttention_1/Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ  
-model/DMGAttention_1/BiasAdd_1/ReadVariableOpReadVariableOp6model_dmgattention_1_biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0ж
model/DMGAttention_1/BiasAdd_1BiasAdd(model/DMGAttention_1/Reshape_23:output:05model/DMGAttention_1/BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Ы
model/DMGAttention_1/stackPack%model/DMGAttention_1/BiasAdd:output:0'model/DMGAttention_1/BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ m
+model/DMGAttention_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : Ш
model/DMGAttention_1/MeanMean#model/DMGAttention_1/stack:output:04model/DMGAttention_1/Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ }
(model/DMGAttention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*model/DMGAttention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*model/DMGAttention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         љ
"model/DMGAttention_1/strided_sliceStridedSlice"model/DMGAttention_1/Mean:output:01model/DMGAttention_1/strided_slice/stack:output:03model/DMGAttention_1/strided_slice/stack_1:output:03model/DMGAttention_1/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_mask
model/DMGAttention_1/EluElu+model/DMGAttention_1/strided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ j
(model/DMGReduce_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model/DMGReduce_1/MeanMean&model/DMGAttention_1/Elu:activations:01model/DMGReduce_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Ђ
,model/DMDense_Hidden_0/MatMul/ReadVariableOpReadVariableOp5model_dmdense_hidden_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype0А
model/DMDense_Hidden_0/MatMulMatMulmodel/DMGReduce_1/Mean:output:04model/DMDense_Hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
-model/DMDense_Hidden_0/BiasAdd/ReadVariableOpReadVariableOp6model_dmdense_hidden_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
model/DMDense_Hidden_0/BiasAddBiasAdd'model/DMDense_Hidden_0/MatMul:product:05model/DMDense_Hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
model/DMDense_Hidden_0/EluElu'model/DMDense_Hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
,model/DMDense_Hidden_1/MatMul/ReadVariableOpReadVariableOp5model_dmdense_hidden_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Й
model/DMDense_Hidden_1/MatMulMatMul(model/DMDense_Hidden_0/Elu:activations:04model/DMDense_Hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
-model/DMDense_Hidden_1/BiasAdd/ReadVariableOpReadVariableOp6model_dmdense_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
model/DMDense_Hidden_1/BiasAddBiasAdd'model/DMDense_Hidden_1/MatMul:product:05model/DMDense_Hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
model/DMDense_Hidden_1/EluElu'model/DMDense_Hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/DMDense_OUT/MatMul/ReadVariableOpReadVariableOp0model_dmdense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Џ
model/DMDense_OUT/MatMulMatMul(model/DMDense_Hidden_1/Elu:activations:0/model/DMDense_OUT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model/DMDense_OUT/BiasAdd/ReadVariableOpReadVariableOp1model_dmdense_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
model/DMDense_OUT/BiasAddBiasAdd"model/DMDense_OUT/MatMul:product:00model/DMDense_OUT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџq
IdentityIdentity"model/DMDense_OUT/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџщ
NoOpNoOp.^model/DMDense_Hidden_0/BiasAdd/ReadVariableOp-^model/DMDense_Hidden_0/MatMul/ReadVariableOp.^model/DMDense_Hidden_1/BiasAdd/ReadVariableOp-^model/DMDense_Hidden_1/MatMul/ReadVariableOp)^model/DMDense_OUT/BiasAdd/ReadVariableOp(^model/DMDense_OUT/MatMul/ReadVariableOp,^model/DMGAttention_0/BiasAdd/ReadVariableOp.^model/DMGAttention_0/BiasAdd_1/ReadVariableOp.^model/DMGAttention_0/transpose/ReadVariableOp0^model/DMGAttention_0/transpose_1/ReadVariableOp0^model/DMGAttention_0/transpose_2/ReadVariableOp0^model/DMGAttention_0/transpose_5/ReadVariableOp0^model/DMGAttention_0/transpose_6/ReadVariableOp0^model/DMGAttention_0/transpose_7/ReadVariableOp,^model/DMGAttention_1/BiasAdd/ReadVariableOp.^model/DMGAttention_1/BiasAdd_1/ReadVariableOp.^model/DMGAttention_1/transpose/ReadVariableOp0^model/DMGAttention_1/transpose_1/ReadVariableOp0^model/DMGAttention_1/transpose_2/ReadVariableOp0^model/DMGAttention_1/transpose_5/ReadVariableOp0^model/DMGAttention_1/transpose_6/ReadVariableOp0^model/DMGAttention_1/transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2^
-model/DMDense_Hidden_0/BiasAdd/ReadVariableOp-model/DMDense_Hidden_0/BiasAdd/ReadVariableOp2\
,model/DMDense_Hidden_0/MatMul/ReadVariableOp,model/DMDense_Hidden_0/MatMul/ReadVariableOp2^
-model/DMDense_Hidden_1/BiasAdd/ReadVariableOp-model/DMDense_Hidden_1/BiasAdd/ReadVariableOp2\
,model/DMDense_Hidden_1/MatMul/ReadVariableOp,model/DMDense_Hidden_1/MatMul/ReadVariableOp2T
(model/DMDense_OUT/BiasAdd/ReadVariableOp(model/DMDense_OUT/BiasAdd/ReadVariableOp2R
'model/DMDense_OUT/MatMul/ReadVariableOp'model/DMDense_OUT/MatMul/ReadVariableOp2Z
+model/DMGAttention_0/BiasAdd/ReadVariableOp+model/DMGAttention_0/BiasAdd/ReadVariableOp2^
-model/DMGAttention_0/BiasAdd_1/ReadVariableOp-model/DMGAttention_0/BiasAdd_1/ReadVariableOp2^
-model/DMGAttention_0/transpose/ReadVariableOp-model/DMGAttention_0/transpose/ReadVariableOp2b
/model/DMGAttention_0/transpose_1/ReadVariableOp/model/DMGAttention_0/transpose_1/ReadVariableOp2b
/model/DMGAttention_0/transpose_2/ReadVariableOp/model/DMGAttention_0/transpose_2/ReadVariableOp2b
/model/DMGAttention_0/transpose_5/ReadVariableOp/model/DMGAttention_0/transpose_5/ReadVariableOp2b
/model/DMGAttention_0/transpose_6/ReadVariableOp/model/DMGAttention_0/transpose_6/ReadVariableOp2b
/model/DMGAttention_0/transpose_7/ReadVariableOp/model/DMGAttention_0/transpose_7/ReadVariableOp2Z
+model/DMGAttention_1/BiasAdd/ReadVariableOp+model/DMGAttention_1/BiasAdd/ReadVariableOp2^
-model/DMGAttention_1/BiasAdd_1/ReadVariableOp-model/DMGAttention_1/BiasAdd_1/ReadVariableOp2^
-model/DMGAttention_1/transpose/ReadVariableOp-model/DMGAttention_1/transpose/ReadVariableOp2b
/model/DMGAttention_1/transpose_1/ReadVariableOp/model/DMGAttention_1/transpose_1/ReadVariableOp2b
/model/DMGAttention_1/transpose_2/ReadVariableOp/model/DMGAttention_1/transpose_2/ReadVariableOp2b
/model/DMGAttention_1/transpose_5/ReadVariableOp/model/DMGAttention_1/transpose_5/ReadVariableOp2b
/model/DMGAttention_1/transpose_6/ReadVariableOp/model/DMGAttention_1/transpose_6/ReadVariableOp2b
/model/DMGAttention_1/transpose_7/ReadVariableOp/model/DMGAttention_1/transpose_7/ReadVariableOp:d `
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
(
_user_specified_nameFeature_Matrix:ok
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
*
_user_specified_nameAdjacency_Matrix
о	

I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51633406
input_tensor0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMulMatMulinput_tensorMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:U Q
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
сЩ
§
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633757

inputs
inputs_11
shape_1_readvariableop_resource:  1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:  2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_1_Attention_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Ч
,DMGAttention_1_Attention_Dropout/dropout/MulMulSoftmax:softmax:07DMGAttention_1_Attention_Dropout/dropout/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџo
.DMGAttention_1_Attention_Dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:ф
EDMGAttention_1_Attention_Dropout/dropout/random_uniform/RandomUniformRandomUniform7DMGAttention_1_Attention_Dropout/dropout/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0|
7DMGAttention_1_Attention_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_1_Attention_Dropout/dropout/GreaterEqualGreaterEqualNDMGAttention_1_Attention_Dropout/dropout/random_uniform/RandomUniform:output:0@DMGAttention_1_Attention_Dropout/dropout/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЧ
-DMGAttention_1_Attention_Dropout/dropout/CastCast9DMGAttention_1_Attention_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџт
.DMGAttention_1_Attention_Dropout/dropout/Mul_1Mul0DMGAttention_1_Attention_Dropout/dropout/Mul:z:01DMGAttention_1_Attention_Dropout/dropout/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџq
,DMGAttention_1_Feature_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Л
*DMGAttention_1_Feature_Dropout/dropout/MulMulReshape_2:output:05DMGAttention_1_Feature_Dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
,DMGAttention_1_Feature_Dropout/dropout/ShapeShapeReshape_2:output:0*
T0*
_output_shapes
:з
CDMGAttention_1_Feature_Dropout/dropout/random_uniform/RandomUniformRandomUniform5DMGAttention_1_Feature_Dropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0z
5DMGAttention_1_Feature_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
3DMGAttention_1_Feature_Dropout/dropout/GreaterEqualGreaterEqualLDMGAttention_1_Feature_Dropout/dropout/random_uniform/RandomUniform:output:0>DMGAttention_1_Feature_Dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ К
+DMGAttention_1_Feature_Dropout/dropout/CastCast7DMGAttention_1_Feature_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ г
,DMGAttention_1_Feature_Dropout/dropout/Mul_1Mul.DMGAttention_1_Feature_Dropout/dropout/Mul:z:0/DMGAttention_1_Feature_Dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
Shape_6Shape2DMGAttention_1_Attention_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numg
Shape_7Shape0DMGAttention_1_Feature_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:
	Reshape_9Reshape2DMGAttention_1_Attention_Dropout/dropout/Mul_1:z:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Є
transpose_4	Transpose0DMGAttention_1_Feature_Dropout/dropout/Mul_1:z:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ =
Shape_8Shapeinputs*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    j

Reshape_12ReshapeinputsReshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:  a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:  n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
0DMGAttention_1_Attention_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Э
.DMGAttention_1_Attention_Dropout/dropout_1/MulMulSoftmax_1:softmax:09DMGAttention_1_Attention_Dropout/dropout_1/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
0DMGAttention_1_Attention_Dropout/dropout_1/ShapeShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:ш
GDMGAttention_1_Attention_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform9DMGAttention_1_Attention_Dropout/dropout_1/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0~
9DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ѕ
7DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqualGreaterEqualPDMGAttention_1_Attention_Dropout/dropout_1/random_uniform/RandomUniform:output:0BDMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЫ
/DMGAttention_1_Attention_Dropout/dropout_1/CastCast;DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџш
0DMGAttention_1_Attention_Dropout/dropout_1/Mul_1Mul2DMGAttention_1_Attention_Dropout/dropout_1/Mul:z:03DMGAttention_1_Attention_Dropout/dropout_1/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_1_Feature_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Р
,DMGAttention_1_Feature_Dropout/dropout_1/MulMulReshape_14:output:07DMGAttention_1_Feature_Dropout/dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ q
.DMGAttention_1_Feature_Dropout/dropout_1/ShapeShapeReshape_14:output:0*
T0*
_output_shapes
:л
EDMGAttention_1_Feature_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform7DMGAttention_1_Feature_Dropout/dropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0|
7DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqualGreaterEqualNDMGAttention_1_Feature_Dropout/dropout_1/random_uniform/RandomUniform:output:0@DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ О
-DMGAttention_1_Feature_Dropout/dropout_1/CastCast9DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ й
.DMGAttention_1_Feature_Dropout/dropout_1/Mul_1Mul0DMGAttention_1_Feature_Dropout/dropout_1/Mul:z:01DMGAttention_1_Feature_Dropout/dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
Shape_14Shape4DMGAttention_1_Attention_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numj
Shape_15Shape2DMGAttention_1_Feature_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:Ё

Reshape_21Reshape4DMGAttention_1_Attention_Dropout/dropout_1/Mul_1:z:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          І
transpose_9	Transpose2DMGAttention_1_Feature_Dropout/dropout_1/Mul_1:z:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЫЃ
Ј
C__inference_model_layer_call_and_return_conditional_losses_51634947
inputs_0
inputs_1@
.dmgattention_0_shape_1_readvariableop_resource:( @
.dmgattention_0_shape_3_readvariableop_resource: @
.dmgattention_0_shape_5_readvariableop_resource: <
.dmgattention_0_biasadd_readvariableop_resource: @
.dmgattention_0_shape_9_readvariableop_resource:( A
/dmgattention_0_shape_11_readvariableop_resource: A
/dmgattention_0_shape_13_readvariableop_resource: >
0dmgattention_0_biasadd_1_readvariableop_resource: @
.dmgattention_1_shape_1_readvariableop_resource:  @
.dmgattention_1_shape_3_readvariableop_resource: @
.dmgattention_1_shape_5_readvariableop_resource: <
.dmgattention_1_biasadd_readvariableop_resource: @
.dmgattention_1_shape_9_readvariableop_resource:  A
/dmgattention_1_shape_11_readvariableop_resource: A
/dmgattention_1_shape_13_readvariableop_resource: >
0dmgattention_1_biasadd_1_readvariableop_resource: A
/dmdense_hidden_0_matmul_readvariableop_resource: >
0dmdense_hidden_0_biasadd_readvariableop_resource:A
/dmdense_hidden_1_matmul_readvariableop_resource:>
0dmdense_hidden_1_biasadd_readvariableop_resource:<
*dmdense_out_matmul_readvariableop_resource:9
+dmdense_out_biasadd_readvariableop_resource:
identityЂ'DMDense_Hidden_0/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_0/MatMul/ReadVariableOpЂ'DMDense_Hidden_1/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_1/MatMul/ReadVariableOpЂ"DMDense_OUT/BiasAdd/ReadVariableOpЂ!DMDense_OUT/MatMul/ReadVariableOpЂ%DMGAttention_0/BiasAdd/ReadVariableOpЂ'DMGAttention_0/BiasAdd_1/ReadVariableOpЂ'DMGAttention_0/transpose/ReadVariableOpЂ)DMGAttention_0/transpose_1/ReadVariableOpЂ)DMGAttention_0/transpose_2/ReadVariableOpЂ)DMGAttention_0/transpose_5/ReadVariableOpЂ)DMGAttention_0/transpose_6/ReadVariableOpЂ)DMGAttention_0/transpose_7/ReadVariableOpЂ%DMGAttention_1/BiasAdd/ReadVariableOpЂ'DMGAttention_1/BiasAdd_1/ReadVariableOpЂ'DMGAttention_1/transpose/ReadVariableOpЂ)DMGAttention_1/transpose_1/ReadVariableOpЂ)DMGAttention_1/transpose_2/ReadVariableOpЂ)DMGAttention_1/transpose_5/ReadVariableOpЂ)DMGAttention_1/transpose_6/ReadVariableOpЂ)DMGAttention_1/transpose_7/ReadVariableOpL
DMGAttention_0/ShapeShapeinputs_0*
T0*
_output_shapes
:o
DMGAttention_0/unstackUnpackDMGAttention_0/Shape:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_1/ReadVariableOpReadVariableOp.dmgattention_0_shape_1_readvariableop_resource*
_output_shapes

:( *
dtype0g
DMGAttention_0/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"(       q
DMGAttention_0/unstack_1UnpackDMGAttention_0/Shape_1:output:0*
T0*
_output_shapes
: : *	
numm
DMGAttention_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   
DMGAttention_0/ReshapeReshapeinputs_0%DMGAttention_0/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(
'DMGAttention_0/transpose/ReadVariableOpReadVariableOp.dmgattention_0_shape_1_readvariableop_resource*
_output_shapes

:( *
dtype0n
DMGAttention_0/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ї
DMGAttention_0/transpose	Transpose/DMGAttention_0/transpose/ReadVariableOp:value:0&DMGAttention_0/transpose/perm:output:0*
T0*
_output_shapes

:( o
DMGAttention_0/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџ
DMGAttention_0/Reshape_1ReshapeDMGAttention_0/transpose:y:0'DMGAttention_0/Reshape_1/shape:output:0*
T0*
_output_shapes

:( 
DMGAttention_0/MatMulMatMulDMGAttention_0/Reshape:output:0!DMGAttention_0/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ b
 DMGAttention_0/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : С
DMGAttention_0/Reshape_2/shapePackDMGAttention_0/unstack:output:0DMGAttention_0/unstack:output:1)DMGAttention_0/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Ќ
DMGAttention_0/Reshape_2ReshapeDMGAttention_0/MatMul:product:0'DMGAttention_0/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ g
DMGAttention_0/Shape_2Shape!DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_2UnpackDMGAttention_0/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_3/ReadVariableOpReadVariableOp.dmgattention_0_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_0/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_0/unstack_3UnpackDMGAttention_0/Shape_3:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_0/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_0/Reshape_3Reshape!DMGAttention_0/Reshape_2:output:0'DMGAttention_0/Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_1/ReadVariableOpReadVariableOp.dmgattention_0_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_1	Transpose1DMGAttention_0/transpose_1/ReadVariableOp:value:0(DMGAttention_0/transpose_1/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_0/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_4ReshapeDMGAttention_0/transpose_1:y:0'DMGAttention_0/Reshape_4/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_1MatMul!DMGAttention_0/Reshape_3:output:0!DMGAttention_0/Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_0/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_0/Reshape_5/shapePack!DMGAttention_0/unstack_2:output:0!DMGAttention_0/unstack_2:output:1)DMGAttention_0/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_0/Reshape_5Reshape!DMGAttention_0/MatMul_1:product:0'DMGAttention_0/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџg
DMGAttention_0/Shape_4Shape!DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_4UnpackDMGAttention_0/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_5/ReadVariableOpReadVariableOp.dmgattention_0_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_0/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_0/unstack_5UnpackDMGAttention_0/Shape_5:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_0/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_0/Reshape_6Reshape!DMGAttention_0/Reshape_2:output:0'DMGAttention_0/Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_2/ReadVariableOpReadVariableOp.dmgattention_0_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_2	Transpose1DMGAttention_0/transpose_2/ReadVariableOp:value:0(DMGAttention_0/transpose_2/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_0/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_7ReshapeDMGAttention_0/transpose_2:y:0'DMGAttention_0/Reshape_7/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_2MatMul!DMGAttention_0/Reshape_6:output:0!DMGAttention_0/Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_0/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_0/Reshape_8/shapePack!DMGAttention_0/unstack_4:output:0!DMGAttention_0/unstack_4:output:1)DMGAttention_0/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_0/Reshape_8Reshape!DMGAttention_0/MatMul_2:product:0'DMGAttention_0/Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
DMGAttention_0/transpose_3	Transpose!DMGAttention_0/Reshape_8:output:0(DMGAttention_0/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџІ
DMGAttention_0/addAddV2!DMGAttention_0/Reshape_5:output:0DMGAttention_0/transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
DMGAttention_0/LeakyRelu	LeakyReluDMGAttention_0/add:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_0/subSubDMGAttention_0/sub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_0/mulMulDMGAttention_0/mul/x:output:0DMGAttention_0/sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
DMGAttention_0/add_1AddV2&DMGAttention_0/LeakyRelu:activations:0DMGAttention_0/mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_0/SoftmaxSoftmaxDMGAttention_0/add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџf
DMGAttention_0/Shape_6Shape DMGAttention_0/Softmax:softmax:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_6UnpackDMGAttention_0/Shape_6:output:0*
T0*
_output_shapes
: : : *	
numg
DMGAttention_0/Shape_7Shape!DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_7UnpackDMGAttention_0/Shape_7:output:0*
T0*
_output_shapes
: : : *	
numk
 DMGAttention_0/Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЂ
DMGAttention_0/Reshape_9/shapePack)DMGAttention_0/Reshape_9/shape/0:output:0!DMGAttention_0/unstack_6:output:2*
N*
T0*
_output_shapes
:Љ
DMGAttention_0/Reshape_9Reshape DMGAttention_0/Softmax:softmax:0'DMGAttention_0/Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
DMGAttention_0/transpose_4	Transpose!DMGAttention_0/Reshape_2:output:0(DMGAttention_0/transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_0/Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЄ
DMGAttention_0/Reshape_10/shapePack!DMGAttention_0/unstack_7:output:1*DMGAttention_0/Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_0/Reshape_10ReshapeDMGAttention_0/transpose_4:y:0(DMGAttention_0/Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
DMGAttention_0/MatMul_3MatMul!DMGAttention_0/Reshape_9:output:0"DMGAttention_0/Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_0/Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ъ
DMGAttention_0/Reshape_11/shapePack!DMGAttention_0/unstack_6:output:0!DMGAttention_0/unstack_6:output:1!DMGAttention_0/unstack_7:output:0*DMGAttention_0/Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_0/Reshape_11Reshape!DMGAttention_0/MatMul_3:product:0(DMGAttention_0/Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
%DMGAttention_0/BiasAdd/ReadVariableOpReadVariableOp.dmgattention_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
DMGAttention_0/BiasAddBiasAdd"DMGAttention_0/Reshape_11:output:0-DMGAttention_0/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ N
DMGAttention_0/Shape_8Shapeinputs_0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_8UnpackDMGAttention_0/Shape_8:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_9/ReadVariableOpReadVariableOp.dmgattention_0_shape_9_readvariableop_resource*
_output_shapes

:( *
dtype0g
DMGAttention_0/Shape_9Const*
_output_shapes
:*
dtype0*
valueB"(       q
DMGAttention_0/unstack_9UnpackDMGAttention_0/Shape_9:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_0/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   
DMGAttention_0/Reshape_12Reshapeinputs_0(DMGAttention_0/Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)DMGAttention_0/transpose_5/ReadVariableOpReadVariableOp.dmgattention_0_shape_9_readvariableop_resource*
_output_shapes

:( *
dtype0p
DMGAttention_0/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_5	Transpose1DMGAttention_0/transpose_5/ReadVariableOp:value:0(DMGAttention_0/transpose_5/perm:output:0*
T0*
_output_shapes

:( p
DMGAttention_0/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџ
DMGAttention_0/Reshape_13ReshapeDMGAttention_0/transpose_5:y:0(DMGAttention_0/Reshape_13/shape:output:0*
T0*
_output_shapes

:( 
DMGAttention_0/MatMul_4MatMul"DMGAttention_0/Reshape_12:output:0"DMGAttention_0/Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c
!DMGAttention_0/Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Ч
DMGAttention_0/Reshape_14/shapePack!DMGAttention_0/unstack_8:output:0!DMGAttention_0/unstack_8:output:1*DMGAttention_0/Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_0/Reshape_14Reshape!DMGAttention_0/MatMul_4:product:0(DMGAttention_0/Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
DMGAttention_0/Shape_10Shape"DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_10Unpack DMGAttention_0/Shape_10:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_0/Shape_11/ReadVariableOpReadVariableOp/dmgattention_0_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_0/Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_0/unstack_11Unpack DMGAttention_0/Shape_11:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_0/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_0/Reshape_15Reshape"DMGAttention_0/Reshape_14:output:0(DMGAttention_0/Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_6/ReadVariableOpReadVariableOp/dmgattention_0_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_6	Transpose1DMGAttention_0/transpose_6/ReadVariableOp:value:0(DMGAttention_0/transpose_6/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_0/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_16ReshapeDMGAttention_0/transpose_6:y:0(DMGAttention_0/Reshape_16/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_5MatMul"DMGAttention_0/Reshape_15:output:0"DMGAttention_0/Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_0/Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_0/Reshape_17/shapePack"DMGAttention_0/unstack_10:output:0"DMGAttention_0/unstack_10:output:1*DMGAttention_0/Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_0/Reshape_17Reshape!DMGAttention_0/MatMul_5:product:0(DMGAttention_0/Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџi
DMGAttention_0/Shape_12Shape"DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_12Unpack DMGAttention_0/Shape_12:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_0/Shape_13/ReadVariableOpReadVariableOp/dmgattention_0_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_0/Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_0/unstack_13Unpack DMGAttention_0/Shape_13:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_0/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_0/Reshape_18Reshape"DMGAttention_0/Reshape_14:output:0(DMGAttention_0/Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_7/ReadVariableOpReadVariableOp/dmgattention_0_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_7	Transpose1DMGAttention_0/transpose_7/ReadVariableOp:value:0(DMGAttention_0/transpose_7/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_0/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_19ReshapeDMGAttention_0/transpose_7:y:0(DMGAttention_0/Reshape_19/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_6MatMul"DMGAttention_0/Reshape_18:output:0"DMGAttention_0/Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_0/Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_0/Reshape_20/shapePack"DMGAttention_0/unstack_12:output:0"DMGAttention_0/unstack_12:output:1*DMGAttention_0/Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_0/Reshape_20Reshape!DMGAttention_0/MatMul_6:product:0(DMGAttention_0/Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
DMGAttention_0/transpose_8	Transpose"DMGAttention_0/Reshape_20:output:0(DMGAttention_0/transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЉ
DMGAttention_0/add_2AddV2"DMGAttention_0/Reshape_17:output:0DMGAttention_0/transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_0/LeakyRelu_1	LeakyReluDMGAttention_0/add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_0/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_0/sub_1SubDMGAttention_0/sub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_0/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_0/mul_1MulDMGAttention_0/mul_1/x:output:0DMGAttention_0/sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
DMGAttention_0/add_3AddV2(DMGAttention_0/LeakyRelu_1:activations:0DMGAttention_0/mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_0/Softmax_1SoftmaxDMGAttention_0/add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџi
DMGAttention_0/Shape_14Shape"DMGAttention_0/Softmax_1:softmax:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_14Unpack DMGAttention_0/Shape_14:output:0*
T0*
_output_shapes
: : : *	
numi
DMGAttention_0/Shape_15Shape"DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_15Unpack DMGAttention_0/Shape_15:output:0*
T0*
_output_shapes
: : : *	
numl
!DMGAttention_0/Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_0/Reshape_21/shapePack*DMGAttention_0/Reshape_21/shape/0:output:0"DMGAttention_0/unstack_14:output:2*
N*
T0*
_output_shapes
:­
DMGAttention_0/Reshape_21Reshape"DMGAttention_0/Softmax_1:softmax:0(DMGAttention_0/Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
DMGAttention_0/transpose_9	Transpose"DMGAttention_0/Reshape_14:output:0(DMGAttention_0/transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_0/Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_0/Reshape_22/shapePack"DMGAttention_0/unstack_15:output:1*DMGAttention_0/Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_0/Reshape_22ReshapeDMGAttention_0/transpose_9:y:0(DMGAttention_0/Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЄ
DMGAttention_0/MatMul_7MatMul"DMGAttention_0/Reshape_21:output:0"DMGAttention_0/Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_0/Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
DMGAttention_0/Reshape_23/shapePack"DMGAttention_0/unstack_14:output:0"DMGAttention_0/unstack_14:output:1"DMGAttention_0/unstack_15:output:0*DMGAttention_0/Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_0/Reshape_23Reshape!DMGAttention_0/MatMul_7:product:0(DMGAttention_0/Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
'DMGAttention_0/BiasAdd_1/ReadVariableOpReadVariableOp0dmgattention_0_biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0Ф
DMGAttention_0/BiasAdd_1BiasAdd"DMGAttention_0/Reshape_23:output:0/DMGAttention_0/BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Й
DMGAttention_0/stackPackDMGAttention_0/BiasAdd:output:0!DMGAttention_0/BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ g
%DMGAttention_0/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : Ж
DMGAttention_0/MeanMeanDMGAttention_0/stack:output:0.DMGAttention_0/Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"DMGAttention_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            y
$DMGAttention_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           y
$DMGAttention_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         л
DMGAttention_0/strided_sliceStridedSliceDMGAttention_0/Mean:output:0+DMGAttention_0/strided_slice/stack:output:0-DMGAttention_0/strided_slice/stack_1:output:0-DMGAttention_0/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_mask
DMGAttention_0/EluElu%DMGAttention_0/strided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ d
DMGAttention_1/ShapeShape DMGAttention_0/Elu:activations:0*
T0*
_output_shapes
:o
DMGAttention_1/unstackUnpackDMGAttention_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_1/ReadVariableOpReadVariableOp.dmgattention_1_shape_1_readvariableop_resource*
_output_shapes

:  *
dtype0g
DMGAttention_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"        q
DMGAttention_1/unstack_1UnpackDMGAttention_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numm
DMGAttention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
DMGAttention_1/ReshapeReshape DMGAttention_0/Elu:activations:0%DMGAttention_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'DMGAttention_1/transpose/ReadVariableOpReadVariableOp.dmgattention_1_shape_1_readvariableop_resource*
_output_shapes

:  *
dtype0n
DMGAttention_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ї
DMGAttention_1/transpose	Transpose/DMGAttention_1/transpose/ReadVariableOp:value:0&DMGAttention_1/transpose/perm:output:0*
T0*
_output_shapes

:  o
DMGAttention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_1ReshapeDMGAttention_1/transpose:y:0'DMGAttention_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:  
DMGAttention_1/MatMulMatMulDMGAttention_1/Reshape:output:0!DMGAttention_1/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ b
 DMGAttention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : С
DMGAttention_1/Reshape_2/shapePackDMGAttention_1/unstack:output:0DMGAttention_1/unstack:output:1)DMGAttention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Ќ
DMGAttention_1/Reshape_2ReshapeDMGAttention_1/MatMul:product:0'DMGAttention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ g
DMGAttention_1/Shape_2Shape!DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_2UnpackDMGAttention_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_3/ReadVariableOpReadVariableOp.dmgattention_1_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_1/unstack_3UnpackDMGAttention_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_1/Reshape_3Reshape!DMGAttention_1/Reshape_2:output:0'DMGAttention_1/Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_1/ReadVariableOpReadVariableOp.dmgattention_1_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_1	Transpose1DMGAttention_1/transpose_1/ReadVariableOp:value:0(DMGAttention_1/transpose_1/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_4ReshapeDMGAttention_1/transpose_1:y:0'DMGAttention_1/Reshape_4/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_1MatMul!DMGAttention_1/Reshape_3:output:0!DMGAttention_1/Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_1/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_1/Reshape_5/shapePack!DMGAttention_1/unstack_2:output:0!DMGAttention_1/unstack_2:output:1)DMGAttention_1/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_1/Reshape_5Reshape!DMGAttention_1/MatMul_1:product:0'DMGAttention_1/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџg
DMGAttention_1/Shape_4Shape!DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_4UnpackDMGAttention_1/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_5/ReadVariableOpReadVariableOp.dmgattention_1_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_1/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_1/unstack_5UnpackDMGAttention_1/Shape_5:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_1/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_1/Reshape_6Reshape!DMGAttention_1/Reshape_2:output:0'DMGAttention_1/Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_2/ReadVariableOpReadVariableOp.dmgattention_1_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_2	Transpose1DMGAttention_1/transpose_2/ReadVariableOp:value:0(DMGAttention_1/transpose_2/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_1/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_7ReshapeDMGAttention_1/transpose_2:y:0'DMGAttention_1/Reshape_7/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_2MatMul!DMGAttention_1/Reshape_6:output:0!DMGAttention_1/Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_1/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_1/Reshape_8/shapePack!DMGAttention_1/unstack_4:output:0!DMGAttention_1/unstack_4:output:1)DMGAttention_1/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_1/Reshape_8Reshape!DMGAttention_1/MatMul_2:product:0'DMGAttention_1/Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
DMGAttention_1/transpose_3	Transpose!DMGAttention_1/Reshape_8:output:0(DMGAttention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџІ
DMGAttention_1/addAddV2!DMGAttention_1/Reshape_5:output:0DMGAttention_1/transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
DMGAttention_1/LeakyRelu	LeakyReluDMGAttention_1/add:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_1/subSubDMGAttention_1/sub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_1/mulMulDMGAttention_1/mul/x:output:0DMGAttention_1/sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
DMGAttention_1/add_1AddV2&DMGAttention_1/LeakyRelu:activations:0DMGAttention_1/mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_1/SoftmaxSoftmaxDMGAttention_1/add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџf
DMGAttention_1/Shape_6Shape DMGAttention_1/Softmax:softmax:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_6UnpackDMGAttention_1/Shape_6:output:0*
T0*
_output_shapes
: : : *	
numg
DMGAttention_1/Shape_7Shape!DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_7UnpackDMGAttention_1/Shape_7:output:0*
T0*
_output_shapes
: : : *	
numk
 DMGAttention_1/Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЂ
DMGAttention_1/Reshape_9/shapePack)DMGAttention_1/Reshape_9/shape/0:output:0!DMGAttention_1/unstack_6:output:2*
N*
T0*
_output_shapes
:Љ
DMGAttention_1/Reshape_9Reshape DMGAttention_1/Softmax:softmax:0'DMGAttention_1/Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
DMGAttention_1/transpose_4	Transpose!DMGAttention_1/Reshape_2:output:0(DMGAttention_1/transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_1/Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЄ
DMGAttention_1/Reshape_10/shapePack!DMGAttention_1/unstack_7:output:1*DMGAttention_1/Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_1/Reshape_10ReshapeDMGAttention_1/transpose_4:y:0(DMGAttention_1/Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
DMGAttention_1/MatMul_3MatMul!DMGAttention_1/Reshape_9:output:0"DMGAttention_1/Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_1/Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ъ
DMGAttention_1/Reshape_11/shapePack!DMGAttention_1/unstack_6:output:0!DMGAttention_1/unstack_6:output:1!DMGAttention_1/unstack_7:output:0*DMGAttention_1/Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_1/Reshape_11Reshape!DMGAttention_1/MatMul_3:product:0(DMGAttention_1/Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
%DMGAttention_1/BiasAdd/ReadVariableOpReadVariableOp.dmgattention_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
DMGAttention_1/BiasAddBiasAdd"DMGAttention_1/Reshape_11:output:0-DMGAttention_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ f
DMGAttention_1/Shape_8Shape DMGAttention_0/Elu:activations:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_8UnpackDMGAttention_1/Shape_8:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_9/ReadVariableOpReadVariableOp.dmgattention_1_shape_9_readvariableop_resource*
_output_shapes

:  *
dtype0g
DMGAttention_1/Shape_9Const*
_output_shapes
:*
dtype0*
valueB"        q
DMGAttention_1/unstack_9UnpackDMGAttention_1/Shape_9:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_1/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ђ
DMGAttention_1/Reshape_12Reshape DMGAttention_0/Elu:activations:0(DMGAttention_1/Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_5/ReadVariableOpReadVariableOp.dmgattention_1_shape_9_readvariableop_resource*
_output_shapes

:  *
dtype0p
DMGAttention_1/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_5	Transpose1DMGAttention_1/transpose_5/ReadVariableOp:value:0(DMGAttention_1/transpose_5/perm:output:0*
T0*
_output_shapes

:  p
DMGAttention_1/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_13ReshapeDMGAttention_1/transpose_5:y:0(DMGAttention_1/Reshape_13/shape:output:0*
T0*
_output_shapes

:  
DMGAttention_1/MatMul_4MatMul"DMGAttention_1/Reshape_12:output:0"DMGAttention_1/Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c
!DMGAttention_1/Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Ч
DMGAttention_1/Reshape_14/shapePack!DMGAttention_1/unstack_8:output:0!DMGAttention_1/unstack_8:output:1*DMGAttention_1/Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_1/Reshape_14Reshape!DMGAttention_1/MatMul_4:product:0(DMGAttention_1/Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
DMGAttention_1/Shape_10Shape"DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_10Unpack DMGAttention_1/Shape_10:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_1/Shape_11/ReadVariableOpReadVariableOp/dmgattention_1_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_1/Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_1/unstack_11Unpack DMGAttention_1/Shape_11:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_1/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_1/Reshape_15Reshape"DMGAttention_1/Reshape_14:output:0(DMGAttention_1/Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_6/ReadVariableOpReadVariableOp/dmgattention_1_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_6	Transpose1DMGAttention_1/transpose_6/ReadVariableOp:value:0(DMGAttention_1/transpose_6/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_1/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_16ReshapeDMGAttention_1/transpose_6:y:0(DMGAttention_1/Reshape_16/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_5MatMul"DMGAttention_1/Reshape_15:output:0"DMGAttention_1/Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_1/Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_1/Reshape_17/shapePack"DMGAttention_1/unstack_10:output:0"DMGAttention_1/unstack_10:output:1*DMGAttention_1/Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_1/Reshape_17Reshape!DMGAttention_1/MatMul_5:product:0(DMGAttention_1/Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџi
DMGAttention_1/Shape_12Shape"DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_12Unpack DMGAttention_1/Shape_12:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_1/Shape_13/ReadVariableOpReadVariableOp/dmgattention_1_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_1/Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_1/unstack_13Unpack DMGAttention_1/Shape_13:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_1/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_1/Reshape_18Reshape"DMGAttention_1/Reshape_14:output:0(DMGAttention_1/Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_7/ReadVariableOpReadVariableOp/dmgattention_1_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_7	Transpose1DMGAttention_1/transpose_7/ReadVariableOp:value:0(DMGAttention_1/transpose_7/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_1/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_19ReshapeDMGAttention_1/transpose_7:y:0(DMGAttention_1/Reshape_19/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_6MatMul"DMGAttention_1/Reshape_18:output:0"DMGAttention_1/Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_1/Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_1/Reshape_20/shapePack"DMGAttention_1/unstack_12:output:0"DMGAttention_1/unstack_12:output:1*DMGAttention_1/Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_1/Reshape_20Reshape!DMGAttention_1/MatMul_6:product:0(DMGAttention_1/Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
DMGAttention_1/transpose_8	Transpose"DMGAttention_1/Reshape_20:output:0(DMGAttention_1/transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЉ
DMGAttention_1/add_2AddV2"DMGAttention_1/Reshape_17:output:0DMGAttention_1/transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_1/LeakyRelu_1	LeakyReluDMGAttention_1/add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_1/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_1/sub_1SubDMGAttention_1/sub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_1/mul_1MulDMGAttention_1/mul_1/x:output:0DMGAttention_1/sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
DMGAttention_1/add_3AddV2(DMGAttention_1/LeakyRelu_1:activations:0DMGAttention_1/mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_1/Softmax_1SoftmaxDMGAttention_1/add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџi
DMGAttention_1/Shape_14Shape"DMGAttention_1/Softmax_1:softmax:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_14Unpack DMGAttention_1/Shape_14:output:0*
T0*
_output_shapes
: : : *	
numi
DMGAttention_1/Shape_15Shape"DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_15Unpack DMGAttention_1/Shape_15:output:0*
T0*
_output_shapes
: : : *	
numl
!DMGAttention_1/Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_1/Reshape_21/shapePack*DMGAttention_1/Reshape_21/shape/0:output:0"DMGAttention_1/unstack_14:output:2*
N*
T0*
_output_shapes
:­
DMGAttention_1/Reshape_21Reshape"DMGAttention_1/Softmax_1:softmax:0(DMGAttention_1/Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
DMGAttention_1/transpose_9	Transpose"DMGAttention_1/Reshape_14:output:0(DMGAttention_1/transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_1/Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_1/Reshape_22/shapePack"DMGAttention_1/unstack_15:output:1*DMGAttention_1/Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_1/Reshape_22ReshapeDMGAttention_1/transpose_9:y:0(DMGAttention_1/Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЄ
DMGAttention_1/MatMul_7MatMul"DMGAttention_1/Reshape_21:output:0"DMGAttention_1/Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_1/Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
DMGAttention_1/Reshape_23/shapePack"DMGAttention_1/unstack_14:output:0"DMGAttention_1/unstack_14:output:1"DMGAttention_1/unstack_15:output:0*DMGAttention_1/Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_1/Reshape_23Reshape!DMGAttention_1/MatMul_7:product:0(DMGAttention_1/Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
'DMGAttention_1/BiasAdd_1/ReadVariableOpReadVariableOp0dmgattention_1_biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0Ф
DMGAttention_1/BiasAdd_1BiasAdd"DMGAttention_1/Reshape_23:output:0/DMGAttention_1/BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Й
DMGAttention_1/stackPackDMGAttention_1/BiasAdd:output:0!DMGAttention_1/BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ g
%DMGAttention_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : Ж
DMGAttention_1/MeanMeanDMGAttention_1/stack:output:0.DMGAttention_1/Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"DMGAttention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            y
$DMGAttention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           y
$DMGAttention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         л
DMGAttention_1/strided_sliceStridedSliceDMGAttention_1/Mean:output:0+DMGAttention_1/strided_slice/stack:output:0-DMGAttention_1/strided_slice/stack_1:output:0-DMGAttention_1/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_mask
DMGAttention_1/EluElu%DMGAttention_1/strided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ d
"DMGReduce_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
DMGReduce_1/MeanMean DMGAttention_1/Elu:activations:0+DMGReduce_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&DMDense_Hidden_0/MatMul/ReadVariableOpReadVariableOp/dmdense_hidden_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
DMDense_Hidden_0/MatMulMatMulDMGReduce_1/Mean:output:0.DMDense_Hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'DMDense_Hidden_0/BiasAdd/ReadVariableOpReadVariableOp0dmdense_hidden_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
DMDense_Hidden_0/BiasAddBiasAdd!DMDense_Hidden_0/MatMul:product:0/DMDense_Hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
DMDense_Hidden_0/EluElu!DMDense_Hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
&DMDense_Hidden_1/MatMul/ReadVariableOpReadVariableOp/dmdense_hidden_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ї
DMDense_Hidden_1/MatMulMatMul"DMDense_Hidden_0/Elu:activations:0.DMDense_Hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'DMDense_Hidden_1/BiasAdd/ReadVariableOpReadVariableOp0dmdense_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
DMDense_Hidden_1/BiasAddBiasAdd!DMDense_Hidden_1/MatMul:product:0/DMDense_Hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
DMDense_Hidden_1/EluElu!DMDense_Hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!DMDense_OUT/MatMul/ReadVariableOpReadVariableOp*dmdense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
DMDense_OUT/MatMulMatMul"DMDense_Hidden_1/Elu:activations:0)DMDense_OUT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
"DMDense_OUT/BiasAdd/ReadVariableOpReadVariableOp+dmdense_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
DMDense_OUT/BiasAddBiasAddDMDense_OUT/MatMul:product:0*DMDense_OUT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџk
IdentityIdentityDMDense_OUT/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџх
NoOpNoOp(^DMDense_Hidden_0/BiasAdd/ReadVariableOp'^DMDense_Hidden_0/MatMul/ReadVariableOp(^DMDense_Hidden_1/BiasAdd/ReadVariableOp'^DMDense_Hidden_1/MatMul/ReadVariableOp#^DMDense_OUT/BiasAdd/ReadVariableOp"^DMDense_OUT/MatMul/ReadVariableOp&^DMGAttention_0/BiasAdd/ReadVariableOp(^DMGAttention_0/BiasAdd_1/ReadVariableOp(^DMGAttention_0/transpose/ReadVariableOp*^DMGAttention_0/transpose_1/ReadVariableOp*^DMGAttention_0/transpose_2/ReadVariableOp*^DMGAttention_0/transpose_5/ReadVariableOp*^DMGAttention_0/transpose_6/ReadVariableOp*^DMGAttention_0/transpose_7/ReadVariableOp&^DMGAttention_1/BiasAdd/ReadVariableOp(^DMGAttention_1/BiasAdd_1/ReadVariableOp(^DMGAttention_1/transpose/ReadVariableOp*^DMGAttention_1/transpose_1/ReadVariableOp*^DMGAttention_1/transpose_2/ReadVariableOp*^DMGAttention_1/transpose_5/ReadVariableOp*^DMGAttention_1/transpose_6/ReadVariableOp*^DMGAttention_1/transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2R
'DMDense_Hidden_0/BiasAdd/ReadVariableOp'DMDense_Hidden_0/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_0/MatMul/ReadVariableOp&DMDense_Hidden_0/MatMul/ReadVariableOp2R
'DMDense_Hidden_1/BiasAdd/ReadVariableOp'DMDense_Hidden_1/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_1/MatMul/ReadVariableOp&DMDense_Hidden_1/MatMul/ReadVariableOp2H
"DMDense_OUT/BiasAdd/ReadVariableOp"DMDense_OUT/BiasAdd/ReadVariableOp2F
!DMDense_OUT/MatMul/ReadVariableOp!DMDense_OUT/MatMul/ReadVariableOp2N
%DMGAttention_0/BiasAdd/ReadVariableOp%DMGAttention_0/BiasAdd/ReadVariableOp2R
'DMGAttention_0/BiasAdd_1/ReadVariableOp'DMGAttention_0/BiasAdd_1/ReadVariableOp2R
'DMGAttention_0/transpose/ReadVariableOp'DMGAttention_0/transpose/ReadVariableOp2V
)DMGAttention_0/transpose_1/ReadVariableOp)DMGAttention_0/transpose_1/ReadVariableOp2V
)DMGAttention_0/transpose_2/ReadVariableOp)DMGAttention_0/transpose_2/ReadVariableOp2V
)DMGAttention_0/transpose_5/ReadVariableOp)DMGAttention_0/transpose_5/ReadVariableOp2V
)DMGAttention_0/transpose_6/ReadVariableOp)DMGAttention_0/transpose_6/ReadVariableOp2V
)DMGAttention_0/transpose_7/ReadVariableOp)DMGAttention_0/transpose_7/ReadVariableOp2N
%DMGAttention_1/BiasAdd/ReadVariableOp%DMGAttention_1/BiasAdd/ReadVariableOp2R
'DMGAttention_1/BiasAdd_1/ReadVariableOp'DMGAttention_1/BiasAdd_1/ReadVariableOp2R
'DMGAttention_1/transpose/ReadVariableOp'DMGAttention_1/transpose/ReadVariableOp2V
)DMGAttention_1/transpose_1/ReadVariableOp)DMGAttention_1/transpose_1/ReadVariableOp2V
)DMGAttention_1/transpose_2/ReadVariableOp)DMGAttention_1/transpose_2/ReadVariableOp2V
)DMGAttention_1/transpose_5/ReadVariableOp)DMGAttention_1/transpose_5/ReadVariableOp2V
)DMGAttention_1/transpose_6/ReadVariableOp)DMGAttention_1/transpose_6/ReadVariableOp2V
)DMGAttention_1/transpose_7/ReadVariableOp)DMGAttention_1/transpose_7/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
яЩ
џ
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635906
inputs_0
inputs_11
shape_1_readvariableop_resource:( 1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:( 2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   f
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:( `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:( h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_0_Attention_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Ч
,DMGAttention_0_Attention_Dropout/dropout/MulMulSoftmax:softmax:07DMGAttention_0_Attention_Dropout/dropout/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџo
.DMGAttention_0_Attention_Dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:ф
EDMGAttention_0_Attention_Dropout/dropout/random_uniform/RandomUniformRandomUniform7DMGAttention_0_Attention_Dropout/dropout/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0|
7DMGAttention_0_Attention_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_0_Attention_Dropout/dropout/GreaterEqualGreaterEqualNDMGAttention_0_Attention_Dropout/dropout/random_uniform/RandomUniform:output:0@DMGAttention_0_Attention_Dropout/dropout/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЧ
-DMGAttention_0_Attention_Dropout/dropout/CastCast9DMGAttention_0_Attention_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџт
.DMGAttention_0_Attention_Dropout/dropout/Mul_1Mul0DMGAttention_0_Attention_Dropout/dropout/Mul:z:01DMGAttention_0_Attention_Dropout/dropout/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџq
,DMGAttention_0_Feature_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Л
*DMGAttention_0_Feature_Dropout/dropout/MulMulReshape_2:output:05DMGAttention_0_Feature_Dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
,DMGAttention_0_Feature_Dropout/dropout/ShapeShapeReshape_2:output:0*
T0*
_output_shapes
:з
CDMGAttention_0_Feature_Dropout/dropout/random_uniform/RandomUniformRandomUniform5DMGAttention_0_Feature_Dropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0z
5DMGAttention_0_Feature_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
3DMGAttention_0_Feature_Dropout/dropout/GreaterEqualGreaterEqualLDMGAttention_0_Feature_Dropout/dropout/random_uniform/RandomUniform:output:0>DMGAttention_0_Feature_Dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ К
+DMGAttention_0_Feature_Dropout/dropout/CastCast7DMGAttention_0_Feature_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ г
,DMGAttention_0_Feature_Dropout/dropout/Mul_1Mul.DMGAttention_0_Feature_Dropout/dropout/Mul:z:0/DMGAttention_0_Feature_Dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
Shape_6Shape2DMGAttention_0_Attention_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numg
Shape_7Shape0DMGAttention_0_Feature_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:
	Reshape_9Reshape2DMGAttention_0_Attention_Dropout/dropout/Mul_1:z:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Є
transpose_4	Transpose0DMGAttention_0_Feature_Dropout/dropout/Mul_1:z:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ?
Shape_8Shapeinputs_0*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   l

Reshape_12Reshapeinputs_0Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:( a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:( n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
0DMGAttention_0_Attention_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Э
.DMGAttention_0_Attention_Dropout/dropout_1/MulMulSoftmax_1:softmax:09DMGAttention_0_Attention_Dropout/dropout_1/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
0DMGAttention_0_Attention_Dropout/dropout_1/ShapeShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:ш
GDMGAttention_0_Attention_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform9DMGAttention_0_Attention_Dropout/dropout_1/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0~
9DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ѕ
7DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqualGreaterEqualPDMGAttention_0_Attention_Dropout/dropout_1/random_uniform/RandomUniform:output:0BDMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЫ
/DMGAttention_0_Attention_Dropout/dropout_1/CastCast;DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџш
0DMGAttention_0_Attention_Dropout/dropout_1/Mul_1Mul2DMGAttention_0_Attention_Dropout/dropout_1/Mul:z:03DMGAttention_0_Attention_Dropout/dropout_1/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_0_Feature_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Р
,DMGAttention_0_Feature_Dropout/dropout_1/MulMulReshape_14:output:07DMGAttention_0_Feature_Dropout/dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ q
.DMGAttention_0_Feature_Dropout/dropout_1/ShapeShapeReshape_14:output:0*
T0*
_output_shapes
:л
EDMGAttention_0_Feature_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform7DMGAttention_0_Feature_Dropout/dropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0|
7DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqualGreaterEqualNDMGAttention_0_Feature_Dropout/dropout_1/random_uniform/RandomUniform:output:0@DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ О
-DMGAttention_0_Feature_Dropout/dropout_1/CastCast9DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ й
.DMGAttention_0_Feature_Dropout/dropout_1/Mul_1Mul0DMGAttention_0_Feature_Dropout/dropout_1/Mul:z:01DMGAttention_0_Feature_Dropout/dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
Shape_14Shape4DMGAttention_0_Attention_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numj
Shape_15Shape2DMGAttention_0_Feature_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:Ё

Reshape_21Reshape4DMGAttention_0_Attention_Dropout/dropout_1/Mul_1:z:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          І
transpose_9	Transpose2DMGAttention_0_Feature_Dropout/dropout_1/Mul_1:z:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Д


N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51633373
input_tensor0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMulMatMulinput_tensorMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:U Q
'
_output_shapes
:џџџџџџџџџ 
&
_user_specified_nameinput_tensor
ь
І
3__inference_DMDense_Hidden_0_layer_call_fn_51636408
input_tensor
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51633373o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:џџџџџџџџџ 
&
_user_specified_nameinput_tensor
Н+
ж	
C__inference_model_layer_call_and_return_conditional_losses_51633413

inputs
inputs_1)
dmgattention_0_51633116:( )
dmgattention_0_51633118: )
dmgattention_0_51633120: %
dmgattention_0_51633122: )
dmgattention_0_51633124:( )
dmgattention_0_51633126: )
dmgattention_0_51633128: %
dmgattention_0_51633130: )
dmgattention_1_51633335:  )
dmgattention_1_51633337: )
dmgattention_1_51633339: %
dmgattention_1_51633341: )
dmgattention_1_51633343:  )
dmgattention_1_51633345: )
dmgattention_1_51633347: %
dmgattention_1_51633349: +
dmdense_hidden_0_51633374: '
dmdense_hidden_0_51633376:+
dmdense_hidden_1_51633391:'
dmdense_hidden_1_51633393:&
dmdense_out_51633407:"
dmdense_out_51633409:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂ&DMGAttention_0/StatefulPartitionedCallЂ&DMGAttention_1/StatefulPartitionedCallі
&DMGAttention_0/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dmgattention_0_51633116dmgattention_0_51633118dmgattention_0_51633120dmgattention_0_51633122dmgattention_0_51633124dmgattention_0_51633126dmgattention_0_51633128dmgattention_0_51633130*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51633115Ц
&DMGAttention_1/StatefulPartitionedCallStatefulPartitionedCall/DMGAttention_0/StatefulPartitionedCall:output:0/DMGAttention_0/StatefulPartitionedCall:output:1dmgattention_1_51633335dmgattention_1_51633337dmgattention_1_51633339dmgattention_1_51633341dmgattention_1_51633343dmgattention_1_51633345dmgattention_1_51633347dmgattention_1_51633349*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633334
DMGReduce_1/PartitionedCallPartitionedCall/DMGAttention_1/StatefulPartitionedCall:output:0/DMGAttention_1/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51633360И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51633374dmdense_hidden_0_51633376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51633373Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51633391dmdense_hidden_1_51633393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51633390Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51633407dmdense_out_51633409*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51633406{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall'^DMGAttention_0/StatefulPartitionedCall'^DMGAttention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2P
&DMGAttention_0/StatefulPartitionedCall&DMGAttention_0/StatefulPartitionedCall2P
&DMGAttention_1/StatefulPartitionedCall&DMGAttention_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ФE

!__inference__traced_save_51636560
file_prefixE
Asavev2_dmgattention_0_dmgattention_0_0_weight_read_readvariableopT
Psavev2_dmgattention_0_dmgattention_0_0_self_attention_weight_read_readvariableopX
Tsavev2_dmgattention_0_dmgattention_0_0_neighbor_attention_weight_read_readvariableopC
?savev2_dmgattention_0_dmgattention_0_0_bias_read_readvariableopE
Asavev2_dmgattention_0_dmgattention_0_1_weight_read_readvariableopT
Psavev2_dmgattention_0_dmgattention_0_1_self_attention_weight_read_readvariableopX
Tsavev2_dmgattention_0_dmgattention_0_1_neighbor_attention_weight_read_readvariableopC
?savev2_dmgattention_0_dmgattention_0_1_bias_read_readvariableopE
Asavev2_dmgattention_1_dmgattention_1_0_weight_read_readvariableopT
Psavev2_dmgattention_1_dmgattention_1_0_self_attention_weight_read_readvariableopX
Tsavev2_dmgattention_1_dmgattention_1_0_neighbor_attention_weight_read_readvariableopC
?savev2_dmgattention_1_dmgattention_1_0_bias_read_readvariableopE
Asavev2_dmgattention_1_dmgattention_1_1_weight_read_readvariableopT
Psavev2_dmgattention_1_dmgattention_1_1_self_attention_weight_read_readvariableopX
Tsavev2_dmgattention_1_dmgattention_1_1_neighbor_attention_weight_read_readvariableopC
?savev2_dmgattention_1_dmgattention_1_1_bias_read_readvariableopG
Csavev2_dmdense_hidden_0_dmdense_hidden_0_weight_read_readvariableopE
Asavev2_dmdense_hidden_0_dmdense_hidden_0_bias_read_readvariableopG
Csavev2_dmdense_hidden_1_dmdense_hidden_1_weight_read_readvariableopE
Asavev2_dmdense_hidden_1_dmdense_hidden_1_bias_read_readvariableop=
9savev2_dmdense_out_dmdense_out_weight_read_readvariableop;
7savev2_dmdense_out_dmdense_out_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*С
valueЗBДBGlayer_with_weights-0/DMGAttention_0_0_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/DMGAttention_0_0_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/DMGAttention_0_0_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/DMGAttention_0_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/DMGAttention_0_1_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/DMGAttention_0_1_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/DMGAttention_0_1_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/DMGAttention_0_1_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/DMGAttention_1_0_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/DMGAttention_1_0_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/DMGAttention_1_0_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-1/DMGAttention_1_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/DMGAttention_1_1_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/DMGAttention_1_1_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/DMGAttention_1_1_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-1/DMGAttention_1_1_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-2/DMDense_Hidden_0_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/DMDense_Hidden_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-3/DMDense_Hidden_1_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-3/DMDense_Hidden_1_bias/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-4/DMDense_OUT_weight/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/DMDense_OUT_bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЃ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_dmgattention_0_dmgattention_0_0_weight_read_readvariableopPsavev2_dmgattention_0_dmgattention_0_0_self_attention_weight_read_readvariableopTsavev2_dmgattention_0_dmgattention_0_0_neighbor_attention_weight_read_readvariableop?savev2_dmgattention_0_dmgattention_0_0_bias_read_readvariableopAsavev2_dmgattention_0_dmgattention_0_1_weight_read_readvariableopPsavev2_dmgattention_0_dmgattention_0_1_self_attention_weight_read_readvariableopTsavev2_dmgattention_0_dmgattention_0_1_neighbor_attention_weight_read_readvariableop?savev2_dmgattention_0_dmgattention_0_1_bias_read_readvariableopAsavev2_dmgattention_1_dmgattention_1_0_weight_read_readvariableopPsavev2_dmgattention_1_dmgattention_1_0_self_attention_weight_read_readvariableopTsavev2_dmgattention_1_dmgattention_1_0_neighbor_attention_weight_read_readvariableop?savev2_dmgattention_1_dmgattention_1_0_bias_read_readvariableopAsavev2_dmgattention_1_dmgattention_1_1_weight_read_readvariableopPsavev2_dmgattention_1_dmgattention_1_1_self_attention_weight_read_readvariableopTsavev2_dmgattention_1_dmgattention_1_1_neighbor_attention_weight_read_readvariableop?savev2_dmgattention_1_dmgattention_1_1_bias_read_readvariableopCsavev2_dmdense_hidden_0_dmdense_hidden_0_weight_read_readvariableopAsavev2_dmdense_hidden_0_dmdense_hidden_0_bias_read_readvariableopCsavev2_dmdense_hidden_1_dmdense_hidden_1_weight_read_readvariableopAsavev2_dmdense_hidden_1_dmdense_hidden_1_bias_read_readvariableop9savev2_dmdense_out_dmdense_out_weight_read_readvariableop7savev2_dmdense_out_dmdense_out_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*с
_input_shapesЯ
Ь: :( : : : :( : : : :  : : : :  : : : : :::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:( :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:( :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
: :$	 

_output_shapes

:  :$
 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
т
Ё
.__inference_DMDense_OUT_layer_call_fn_51636448
input_tensor
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51633406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
Ј
ъ
(__inference_model_layer_call_fn_51633460
feature_matrix
adjacency_matrix
unknown:( 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:( 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallfeature_matrixadjacency_matrixunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51633413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
(
_user_specified_nameFeature_Matrix:ok
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
*
_user_specified_nameAdjacency_Matrix
яЩ
џ
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636386
inputs_0
inputs_11
shape_1_readvariableop_resource:  1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:  2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    f
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_1_Attention_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Ч
,DMGAttention_1_Attention_Dropout/dropout/MulMulSoftmax:softmax:07DMGAttention_1_Attention_Dropout/dropout/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџo
.DMGAttention_1_Attention_Dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:ф
EDMGAttention_1_Attention_Dropout/dropout/random_uniform/RandomUniformRandomUniform7DMGAttention_1_Attention_Dropout/dropout/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0|
7DMGAttention_1_Attention_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_1_Attention_Dropout/dropout/GreaterEqualGreaterEqualNDMGAttention_1_Attention_Dropout/dropout/random_uniform/RandomUniform:output:0@DMGAttention_1_Attention_Dropout/dropout/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЧ
-DMGAttention_1_Attention_Dropout/dropout/CastCast9DMGAttention_1_Attention_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџт
.DMGAttention_1_Attention_Dropout/dropout/Mul_1Mul0DMGAttention_1_Attention_Dropout/dropout/Mul:z:01DMGAttention_1_Attention_Dropout/dropout/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџq
,DMGAttention_1_Feature_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Л
*DMGAttention_1_Feature_Dropout/dropout/MulMulReshape_2:output:05DMGAttention_1_Feature_Dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
,DMGAttention_1_Feature_Dropout/dropout/ShapeShapeReshape_2:output:0*
T0*
_output_shapes
:з
CDMGAttention_1_Feature_Dropout/dropout/random_uniform/RandomUniformRandomUniform5DMGAttention_1_Feature_Dropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0z
5DMGAttention_1_Feature_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
3DMGAttention_1_Feature_Dropout/dropout/GreaterEqualGreaterEqualLDMGAttention_1_Feature_Dropout/dropout/random_uniform/RandomUniform:output:0>DMGAttention_1_Feature_Dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ К
+DMGAttention_1_Feature_Dropout/dropout/CastCast7DMGAttention_1_Feature_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ г
,DMGAttention_1_Feature_Dropout/dropout/Mul_1Mul.DMGAttention_1_Feature_Dropout/dropout/Mul:z:0/DMGAttention_1_Feature_Dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
Shape_6Shape2DMGAttention_1_Attention_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numg
Shape_7Shape0DMGAttention_1_Feature_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:
	Reshape_9Reshape2DMGAttention_1_Attention_Dropout/dropout/Mul_1:z:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Є
transpose_4	Transpose0DMGAttention_1_Feature_Dropout/dropout/Mul_1:z:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ?
Shape_8Shapeinputs_0*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    l

Reshape_12Reshapeinputs_0Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:  a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:  n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
0DMGAttention_1_Attention_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Э
.DMGAttention_1_Attention_Dropout/dropout_1/MulMulSoftmax_1:softmax:09DMGAttention_1_Attention_Dropout/dropout_1/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
0DMGAttention_1_Attention_Dropout/dropout_1/ShapeShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:ш
GDMGAttention_1_Attention_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform9DMGAttention_1_Attention_Dropout/dropout_1/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0~
9DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ѕ
7DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqualGreaterEqualPDMGAttention_1_Attention_Dropout/dropout_1/random_uniform/RandomUniform:output:0BDMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЫ
/DMGAttention_1_Attention_Dropout/dropout_1/CastCast;DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџш
0DMGAttention_1_Attention_Dropout/dropout_1/Mul_1Mul2DMGAttention_1_Attention_Dropout/dropout_1/Mul:z:03DMGAttention_1_Attention_Dropout/dropout_1/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_1_Feature_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Р
,DMGAttention_1_Feature_Dropout/dropout_1/MulMulReshape_14:output:07DMGAttention_1_Feature_Dropout/dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ q
.DMGAttention_1_Feature_Dropout/dropout_1/ShapeShapeReshape_14:output:0*
T0*
_output_shapes
:л
EDMGAttention_1_Feature_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform7DMGAttention_1_Feature_Dropout/dropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0|
7DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqualGreaterEqualNDMGAttention_1_Feature_Dropout/dropout_1/random_uniform/RandomUniform:output:0@DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ О
-DMGAttention_1_Feature_Dropout/dropout_1/CastCast9DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ й
.DMGAttention_1_Feature_Dropout/dropout_1/Mul_1Mul0DMGAttention_1_Feature_Dropout/dropout_1/Mul:z:01DMGAttention_1_Feature_Dropout/dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
Shape_14Shape4DMGAttention_1_Attention_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numj
Shape_15Shape2DMGAttention_1_Feature_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:Ё

Reshape_21Reshape4DMGAttention_1_Attention_Dropout/dropout_1/Mul_1:z:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          І
transpose_9	Transpose2DMGAttention_1_Feature_Dropout/dropout_1/Mul_1:z:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
я+
ц	
C__inference_model_layer_call_and_return_conditional_losses_51634380
feature_matrix
adjacency_matrix)
dmgattention_0_51634327:( )
dmgattention_0_51634329: )
dmgattention_0_51634331: %
dmgattention_0_51634333: )
dmgattention_0_51634335:( )
dmgattention_0_51634337: )
dmgattention_0_51634339: %
dmgattention_0_51634341: )
dmgattention_1_51634345:  )
dmgattention_1_51634347: )
dmgattention_1_51634349: %
dmgattention_1_51634351: )
dmgattention_1_51634353:  )
dmgattention_1_51634355: )
dmgattention_1_51634357: %
dmgattention_1_51634359: +
dmdense_hidden_0_51634364: '
dmdense_hidden_0_51634366:+
dmdense_hidden_1_51634369:'
dmdense_hidden_1_51634371:&
dmdense_out_51634374:"
dmdense_out_51634376:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂ&DMGAttention_0/StatefulPartitionedCallЂ&DMGAttention_1/StatefulPartitionedCall
&DMGAttention_0/StatefulPartitionedCallStatefulPartitionedCallfeature_matrixadjacency_matrixdmgattention_0_51634327dmgattention_0_51634329dmgattention_0_51634331dmgattention_0_51634333dmgattention_0_51634335dmgattention_0_51634337dmgattention_0_51634339dmgattention_0_51634341*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51634038Ц
&DMGAttention_1/StatefulPartitionedCallStatefulPartitionedCall/DMGAttention_0/StatefulPartitionedCall:output:0/DMGAttention_0/StatefulPartitionedCall:output:1dmgattention_1_51634345dmgattention_1_51634347dmgattention_1_51634349dmgattention_1_51634351dmgattention_1_51634353dmgattention_1_51634355dmgattention_1_51634357dmgattention_1_51634359*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633757
DMGReduce_1/PartitionedCallPartitionedCall/DMGAttention_1/StatefulPartitionedCall:output:0/DMGAttention_1/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51633360И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51634364dmdense_hidden_0_51634366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51633373Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51634369dmdense_hidden_1_51634371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51633390Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51634374dmdense_out_51634376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51633406{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall'^DMGAttention_0/StatefulPartitionedCall'^DMGAttention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2P
&DMGAttention_0/StatefulPartitionedCall&DMGAttention_0/StatefulPartitionedCall2P
&DMGAttention_1/StatefulPartitionedCall&DMGAttention_1/StatefulPartitionedCall:d `
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
(
_user_specified_nameFeature_Matrix:ok
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
*
_user_specified_nameAdjacency_Matrix
і
Z
.__inference_DMGReduce_1_layer_call_fn_51636392
inputs_0
inputs_1
identityХ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51633360`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
ь
І
3__inference_DMDense_Hidden_1_layer_call_fn_51636428
input_tensor
unknown:
	unknown_0:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51633390o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor

u
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51636399
inputs_0
inputs_1
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
MeanMeaninputs_0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Д


N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51633390
input_tensor0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMulMatMulinput_tensorMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:U Q
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
ћ
s
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51633360

inputs
inputs_1
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ьs

$__inference__traced_restore_51636648
file_prefixI
7assignvariableop_dmgattention_0_dmgattention_0_0_weight:( Z
Hassignvariableop_1_dmgattention_0_dmgattention_0_0_self_attention_weight: ^
Lassignvariableop_2_dmgattention_0_dmgattention_0_0_neighbor_attention_weight: E
7assignvariableop_3_dmgattention_0_dmgattention_0_0_bias: K
9assignvariableop_4_dmgattention_0_dmgattention_0_1_weight:( Z
Hassignvariableop_5_dmgattention_0_dmgattention_0_1_self_attention_weight: ^
Lassignvariableop_6_dmgattention_0_dmgattention_0_1_neighbor_attention_weight: E
7assignvariableop_7_dmgattention_0_dmgattention_0_1_bias: K
9assignvariableop_8_dmgattention_1_dmgattention_1_0_weight:  Z
Hassignvariableop_9_dmgattention_1_dmgattention_1_0_self_attention_weight: _
Massignvariableop_10_dmgattention_1_dmgattention_1_0_neighbor_attention_weight: F
8assignvariableop_11_dmgattention_1_dmgattention_1_0_bias: L
:assignvariableop_12_dmgattention_1_dmgattention_1_1_weight:  [
Iassignvariableop_13_dmgattention_1_dmgattention_1_1_self_attention_weight: _
Massignvariableop_14_dmgattention_1_dmgattention_1_1_neighbor_attention_weight: F
8assignvariableop_15_dmgattention_1_dmgattention_1_1_bias: N
<assignvariableop_16_dmdense_hidden_0_dmdense_hidden_0_weight: H
:assignvariableop_17_dmdense_hidden_0_dmdense_hidden_0_bias:N
<assignvariableop_18_dmdense_hidden_1_dmdense_hidden_1_weight:H
:assignvariableop_19_dmdense_hidden_1_dmdense_hidden_1_bias:D
2assignvariableop_20_dmdense_out_dmdense_out_weight:>
0assignvariableop_21_dmdense_out_dmdense_out_bias:%
assignvariableop_22_total_1: %
assignvariableop_23_count_1: #
assignvariableop_24_total: #
assignvariableop_25_count: 
identity_27ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*С
valueЗBДBGlayer_with_weights-0/DMGAttention_0_0_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/DMGAttention_0_0_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/DMGAttention_0_0_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/DMGAttention_0_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/DMGAttention_0_1_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/DMGAttention_0_1_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/DMGAttention_0_1_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-0/DMGAttention_0_1_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/DMGAttention_1_0_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/DMGAttention_1_0_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/DMGAttention_1_0_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-1/DMGAttention_1_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/DMGAttention_1_1_weight/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/DMGAttention_1_1_self_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/DMGAttention_1_1_neighbor_attention_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-1/DMGAttention_1_1_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-2/DMDense_Hidden_0_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/DMDense_Hidden_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-3/DMDense_Hidden_1_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-3/DMDense_Hidden_1_bias/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-4/DMDense_OUT_weight/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/DMDense_OUT_bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHІ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B І
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOpAssignVariableOp7assignvariableop_dmgattention_0_dmgattention_0_0_weightIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOpHassignvariableop_1_dmgattention_0_dmgattention_0_0_self_attention_weightIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_2AssignVariableOpLassignvariableop_2_dmgattention_0_dmgattention_0_0_neighbor_attention_weightIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_3AssignVariableOp7assignvariableop_3_dmgattention_0_dmgattention_0_0_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_4AssignVariableOp9assignvariableop_4_dmgattention_0_dmgattention_0_1_weightIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOpHassignvariableop_5_dmgattention_0_dmgattention_0_1_self_attention_weightIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpLassignvariableop_6_dmgattention_0_dmgattention_0_1_neighbor_attention_weightIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_7AssignVariableOp7assignvariableop_7_dmgattention_0_dmgattention_0_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_8AssignVariableOp9assignvariableop_8_dmgattention_1_dmgattention_1_0_weightIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOpHassignvariableop_9_dmgattention_1_dmgattention_1_0_self_attention_weightIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_10AssignVariableOpMassignvariableop_10_dmgattention_1_dmgattention_1_0_neighbor_attention_weightIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_11AssignVariableOp8assignvariableop_11_dmgattention_1_dmgattention_1_0_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_12AssignVariableOp:assignvariableop_12_dmgattention_1_dmgattention_1_1_weightIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpIassignvariableop_13_dmgattention_1_dmgattention_1_1_self_attention_weightIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_14AssignVariableOpMassignvariableop_14_dmgattention_1_dmgattention_1_1_neighbor_attention_weightIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_15AssignVariableOp8assignvariableop_15_dmgattention_1_dmgattention_1_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_16AssignVariableOp<assignvariableop_16_dmdense_hidden_0_dmdense_hidden_0_weightIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_17AssignVariableOp:assignvariableop_17_dmdense_hidden_0_dmdense_hidden_0_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_18AssignVariableOp<assignvariableop_18_dmdense_hidden_1_dmdense_hidden_1_weightIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_19AssignVariableOp:assignvariableop_19_dmdense_hidden_1_dmdense_hidden_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_20AssignVariableOp2assignvariableop_20_dmdense_out_dmdense_out_weightIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_21AssignVariableOp0assignvariableop_21_dmdense_out_dmdense_out_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: ј
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
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
я
Ј
C__inference_model_layer_call_and_return_conditional_losses_51635426
inputs_0
inputs_1@
.dmgattention_0_shape_1_readvariableop_resource:( @
.dmgattention_0_shape_3_readvariableop_resource: @
.dmgattention_0_shape_5_readvariableop_resource: <
.dmgattention_0_biasadd_readvariableop_resource: @
.dmgattention_0_shape_9_readvariableop_resource:( A
/dmgattention_0_shape_11_readvariableop_resource: A
/dmgattention_0_shape_13_readvariableop_resource: >
0dmgattention_0_biasadd_1_readvariableop_resource: @
.dmgattention_1_shape_1_readvariableop_resource:  @
.dmgattention_1_shape_3_readvariableop_resource: @
.dmgattention_1_shape_5_readvariableop_resource: <
.dmgattention_1_biasadd_readvariableop_resource: @
.dmgattention_1_shape_9_readvariableop_resource:  A
/dmgattention_1_shape_11_readvariableop_resource: A
/dmgattention_1_shape_13_readvariableop_resource: >
0dmgattention_1_biasadd_1_readvariableop_resource: A
/dmdense_hidden_0_matmul_readvariableop_resource: >
0dmdense_hidden_0_biasadd_readvariableop_resource:A
/dmdense_hidden_1_matmul_readvariableop_resource:>
0dmdense_hidden_1_biasadd_readvariableop_resource:<
*dmdense_out_matmul_readvariableop_resource:9
+dmdense_out_biasadd_readvariableop_resource:
identityЂ'DMDense_Hidden_0/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_0/MatMul/ReadVariableOpЂ'DMDense_Hidden_1/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_1/MatMul/ReadVariableOpЂ"DMDense_OUT/BiasAdd/ReadVariableOpЂ!DMDense_OUT/MatMul/ReadVariableOpЂ%DMGAttention_0/BiasAdd/ReadVariableOpЂ'DMGAttention_0/BiasAdd_1/ReadVariableOpЂ'DMGAttention_0/transpose/ReadVariableOpЂ)DMGAttention_0/transpose_1/ReadVariableOpЂ)DMGAttention_0/transpose_2/ReadVariableOpЂ)DMGAttention_0/transpose_5/ReadVariableOpЂ)DMGAttention_0/transpose_6/ReadVariableOpЂ)DMGAttention_0/transpose_7/ReadVariableOpЂ%DMGAttention_1/BiasAdd/ReadVariableOpЂ'DMGAttention_1/BiasAdd_1/ReadVariableOpЂ'DMGAttention_1/transpose/ReadVariableOpЂ)DMGAttention_1/transpose_1/ReadVariableOpЂ)DMGAttention_1/transpose_2/ReadVariableOpЂ)DMGAttention_1/transpose_5/ReadVariableOpЂ)DMGAttention_1/transpose_6/ReadVariableOpЂ)DMGAttention_1/transpose_7/ReadVariableOpL
DMGAttention_0/ShapeShapeinputs_0*
T0*
_output_shapes
:o
DMGAttention_0/unstackUnpackDMGAttention_0/Shape:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_1/ReadVariableOpReadVariableOp.dmgattention_0_shape_1_readvariableop_resource*
_output_shapes

:( *
dtype0g
DMGAttention_0/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"(       q
DMGAttention_0/unstack_1UnpackDMGAttention_0/Shape_1:output:0*
T0*
_output_shapes
: : *	
numm
DMGAttention_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   
DMGAttention_0/ReshapeReshapeinputs_0%DMGAttention_0/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(
'DMGAttention_0/transpose/ReadVariableOpReadVariableOp.dmgattention_0_shape_1_readvariableop_resource*
_output_shapes

:( *
dtype0n
DMGAttention_0/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ї
DMGAttention_0/transpose	Transpose/DMGAttention_0/transpose/ReadVariableOp:value:0&DMGAttention_0/transpose/perm:output:0*
T0*
_output_shapes

:( o
DMGAttention_0/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџ
DMGAttention_0/Reshape_1ReshapeDMGAttention_0/transpose:y:0'DMGAttention_0/Reshape_1/shape:output:0*
T0*
_output_shapes

:( 
DMGAttention_0/MatMulMatMulDMGAttention_0/Reshape:output:0!DMGAttention_0/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ b
 DMGAttention_0/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : С
DMGAttention_0/Reshape_2/shapePackDMGAttention_0/unstack:output:0DMGAttention_0/unstack:output:1)DMGAttention_0/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Ќ
DMGAttention_0/Reshape_2ReshapeDMGAttention_0/MatMul:product:0'DMGAttention_0/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ g
DMGAttention_0/Shape_2Shape!DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_2UnpackDMGAttention_0/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_3/ReadVariableOpReadVariableOp.dmgattention_0_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_0/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_0/unstack_3UnpackDMGAttention_0/Shape_3:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_0/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_0/Reshape_3Reshape!DMGAttention_0/Reshape_2:output:0'DMGAttention_0/Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_1/ReadVariableOpReadVariableOp.dmgattention_0_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_1	Transpose1DMGAttention_0/transpose_1/ReadVariableOp:value:0(DMGAttention_0/transpose_1/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_0/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_4ReshapeDMGAttention_0/transpose_1:y:0'DMGAttention_0/Reshape_4/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_1MatMul!DMGAttention_0/Reshape_3:output:0!DMGAttention_0/Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_0/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_0/Reshape_5/shapePack!DMGAttention_0/unstack_2:output:0!DMGAttention_0/unstack_2:output:1)DMGAttention_0/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_0/Reshape_5Reshape!DMGAttention_0/MatMul_1:product:0'DMGAttention_0/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџg
DMGAttention_0/Shape_4Shape!DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_4UnpackDMGAttention_0/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_5/ReadVariableOpReadVariableOp.dmgattention_0_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_0/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_0/unstack_5UnpackDMGAttention_0/Shape_5:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_0/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_0/Reshape_6Reshape!DMGAttention_0/Reshape_2:output:0'DMGAttention_0/Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_2/ReadVariableOpReadVariableOp.dmgattention_0_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_2	Transpose1DMGAttention_0/transpose_2/ReadVariableOp:value:0(DMGAttention_0/transpose_2/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_0/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_7ReshapeDMGAttention_0/transpose_2:y:0'DMGAttention_0/Reshape_7/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_2MatMul!DMGAttention_0/Reshape_6:output:0!DMGAttention_0/Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_0/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_0/Reshape_8/shapePack!DMGAttention_0/unstack_4:output:0!DMGAttention_0/unstack_4:output:1)DMGAttention_0/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_0/Reshape_8Reshape!DMGAttention_0/MatMul_2:product:0'DMGAttention_0/Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
DMGAttention_0/transpose_3	Transpose!DMGAttention_0/Reshape_8:output:0(DMGAttention_0/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџІ
DMGAttention_0/addAddV2!DMGAttention_0/Reshape_5:output:0DMGAttention_0/transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
DMGAttention_0/LeakyRelu	LeakyReluDMGAttention_0/add:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_0/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_0/subSubDMGAttention_0/sub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_0/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_0/mulMulDMGAttention_0/mul/x:output:0DMGAttention_0/sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
DMGAttention_0/add_1AddV2&DMGAttention_0/LeakyRelu:activations:0DMGAttention_0/mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_0/SoftmaxSoftmaxDMGAttention_0/add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?є
;DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/MulMul DMGAttention_0/Softmax:softmax:0FDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/ShapeShape DMGAttention_0/Softmax:softmax:0*
T0*
_output_shapes
:
TDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/random_uniform/RandomUniformRandomUniformFDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0
FDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ь
DDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/GreaterEqualGreaterEqual]DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/random_uniform/RandomUniform:output:0ODMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџх
<DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/CastCastHDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/Mul_1Mul?DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/Mul:z:0@DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
;DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?ш
9DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/MulMul!DMGAttention_0/Reshape_2:output:0DDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
;DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/ShapeShape!DMGAttention_0/Reshape_2:output:0*
T0*
_output_shapes
:ѕ
RDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/random_uniform/RandomUniformRandomUniformDDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0
DDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Н
BDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/GreaterEqualGreaterEqual[DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/random_uniform/RandomUniform:output:0MDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ и
:DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/CastCastFDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
;DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/Mul_1Mul=DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/Mul:z:0>DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGAttention_0/Shape_6ShapeADMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_6UnpackDMGAttention_0/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num
DMGAttention_0/Shape_7Shape?DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_7UnpackDMGAttention_0/Shape_7:output:0*
T0*
_output_shapes
: : : *	
numk
 DMGAttention_0/Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЂ
DMGAttention_0/Reshape_9/shapePack)DMGAttention_0/Reshape_9/shape/0:output:0!DMGAttention_0/unstack_6:output:2*
N*
T0*
_output_shapes
:Ъ
DMGAttention_0/Reshape_9ReshapeADMGAttention_0/DMGAttention_0_Attention_Dropout/dropout/Mul_1:z:0'DMGAttention_0/Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          б
DMGAttention_0/transpose_4	Transpose?DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout/Mul_1:z:0(DMGAttention_0/transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_0/Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЄ
DMGAttention_0/Reshape_10/shapePack!DMGAttention_0/unstack_7:output:1*DMGAttention_0/Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_0/Reshape_10ReshapeDMGAttention_0/transpose_4:y:0(DMGAttention_0/Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
DMGAttention_0/MatMul_3MatMul!DMGAttention_0/Reshape_9:output:0"DMGAttention_0/Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_0/Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ъ
DMGAttention_0/Reshape_11/shapePack!DMGAttention_0/unstack_6:output:0!DMGAttention_0/unstack_6:output:1!DMGAttention_0/unstack_7:output:0*DMGAttention_0/Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_0/Reshape_11Reshape!DMGAttention_0/MatMul_3:product:0(DMGAttention_0/Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
%DMGAttention_0/BiasAdd/ReadVariableOpReadVariableOp.dmgattention_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
DMGAttention_0/BiasAddBiasAdd"DMGAttention_0/Reshape_11:output:0-DMGAttention_0/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ N
DMGAttention_0/Shape_8Shapeinputs_0*
T0*
_output_shapes
:s
DMGAttention_0/unstack_8UnpackDMGAttention_0/Shape_8:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_0/Shape_9/ReadVariableOpReadVariableOp.dmgattention_0_shape_9_readvariableop_resource*
_output_shapes

:( *
dtype0g
DMGAttention_0/Shape_9Const*
_output_shapes
:*
dtype0*
valueB"(       q
DMGAttention_0/unstack_9UnpackDMGAttention_0/Shape_9:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_0/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   
DMGAttention_0/Reshape_12Reshapeinputs_0(DMGAttention_0/Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(
)DMGAttention_0/transpose_5/ReadVariableOpReadVariableOp.dmgattention_0_shape_9_readvariableop_resource*
_output_shapes

:( *
dtype0p
DMGAttention_0/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_5	Transpose1DMGAttention_0/transpose_5/ReadVariableOp:value:0(DMGAttention_0/transpose_5/perm:output:0*
T0*
_output_shapes

:( p
DMGAttention_0/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџ
DMGAttention_0/Reshape_13ReshapeDMGAttention_0/transpose_5:y:0(DMGAttention_0/Reshape_13/shape:output:0*
T0*
_output_shapes

:( 
DMGAttention_0/MatMul_4MatMul"DMGAttention_0/Reshape_12:output:0"DMGAttention_0/Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c
!DMGAttention_0/Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Ч
DMGAttention_0/Reshape_14/shapePack!DMGAttention_0/unstack_8:output:0!DMGAttention_0/unstack_8:output:1*DMGAttention_0/Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_0/Reshape_14Reshape!DMGAttention_0/MatMul_4:product:0(DMGAttention_0/Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
DMGAttention_0/Shape_10Shape"DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_10Unpack DMGAttention_0/Shape_10:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_0/Shape_11/ReadVariableOpReadVariableOp/dmgattention_0_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_0/Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_0/unstack_11Unpack DMGAttention_0/Shape_11:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_0/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_0/Reshape_15Reshape"DMGAttention_0/Reshape_14:output:0(DMGAttention_0/Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_6/ReadVariableOpReadVariableOp/dmgattention_0_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_6	Transpose1DMGAttention_0/transpose_6/ReadVariableOp:value:0(DMGAttention_0/transpose_6/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_0/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_16ReshapeDMGAttention_0/transpose_6:y:0(DMGAttention_0/Reshape_16/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_5MatMul"DMGAttention_0/Reshape_15:output:0"DMGAttention_0/Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_0/Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_0/Reshape_17/shapePack"DMGAttention_0/unstack_10:output:0"DMGAttention_0/unstack_10:output:1*DMGAttention_0/Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_0/Reshape_17Reshape!DMGAttention_0/MatMul_5:product:0(DMGAttention_0/Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџi
DMGAttention_0/Shape_12Shape"DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_12Unpack DMGAttention_0/Shape_12:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_0/Shape_13/ReadVariableOpReadVariableOp/dmgattention_0_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_0/Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_0/unstack_13Unpack DMGAttention_0/Shape_13:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_0/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_0/Reshape_18Reshape"DMGAttention_0/Reshape_14:output:0(DMGAttention_0/Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_0/transpose_7/ReadVariableOpReadVariableOp/dmgattention_0_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_0/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_0/transpose_7	Transpose1DMGAttention_0/transpose_7/ReadVariableOp:value:0(DMGAttention_0/transpose_7/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_0/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_0/Reshape_19ReshapeDMGAttention_0/transpose_7:y:0(DMGAttention_0/Reshape_19/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_0/MatMul_6MatMul"DMGAttention_0/Reshape_18:output:0"DMGAttention_0/Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_0/Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_0/Reshape_20/shapePack"DMGAttention_0/unstack_12:output:0"DMGAttention_0/unstack_12:output:1*DMGAttention_0/Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_0/Reshape_20Reshape!DMGAttention_0/MatMul_6:product:0(DMGAttention_0/Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
DMGAttention_0/transpose_8	Transpose"DMGAttention_0/Reshape_20:output:0(DMGAttention_0/transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЉ
DMGAttention_0/add_2AddV2"DMGAttention_0/Reshape_17:output:0DMGAttention_0/transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_0/LeakyRelu_1	LeakyReluDMGAttention_0/add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_0/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_0/sub_1SubDMGAttention_0/sub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_0/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_0/mul_1MulDMGAttention_0/mul_1/x:output:0DMGAttention_0/sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
DMGAttention_0/add_3AddV2(DMGAttention_0/LeakyRelu_1:activations:0DMGAttention_0/mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_0/Softmax_1SoftmaxDMGAttention_0/add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
?DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?њ
=DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/MulMul"DMGAttention_0/Softmax_1:softmax:0HDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
?DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/ShapeShape"DMGAttention_0/Softmax_1:softmax:0*
T0*
_output_shapes
:
VDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/random_uniform/RandomUniformRandomUniformHDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0
HDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<в
FDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqualGreaterEqual_DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/random_uniform/RandomUniform:output:0QDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџщ
>DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/CastCastJDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
?DMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/Mul_1MulADMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/Mul:z:0BDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?э
;DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/MulMul"DMGAttention_0/Reshape_14:output:0FDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
=DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/ShapeShape"DMGAttention_0/Reshape_14:output:0*
T0*
_output_shapes
:љ
TDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/random_uniform/RandomUniformRandomUniformFDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0
FDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<У
DDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqualGreaterEqual]DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/random_uniform/RandomUniform:output:0ODMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ м
<DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/CastCastHDMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
=DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/Mul_1Mul?DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/Mul:z:0@DMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGAttention_0/Shape_14ShapeCDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_14Unpack DMGAttention_0/Shape_14:output:0*
T0*
_output_shapes
: : : *	
num
DMGAttention_0/Shape_15ShapeADMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:u
DMGAttention_0/unstack_15Unpack DMGAttention_0/Shape_15:output:0*
T0*
_output_shapes
: : : *	
numl
!DMGAttention_0/Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_0/Reshape_21/shapePack*DMGAttention_0/Reshape_21/shape/0:output:0"DMGAttention_0/unstack_14:output:2*
N*
T0*
_output_shapes
:Ю
DMGAttention_0/Reshape_21ReshapeCDMGAttention_0/DMGAttention_0_Attention_Dropout/dropout_1/Mul_1:z:0(DMGAttention_0/Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_0/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
DMGAttention_0/transpose_9	TransposeADMGAttention_0/DMGAttention_0_Feature_Dropout/dropout_1/Mul_1:z:0(DMGAttention_0/transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_0/Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_0/Reshape_22/shapePack"DMGAttention_0/unstack_15:output:1*DMGAttention_0/Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_0/Reshape_22ReshapeDMGAttention_0/transpose_9:y:0(DMGAttention_0/Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЄ
DMGAttention_0/MatMul_7MatMul"DMGAttention_0/Reshape_21:output:0"DMGAttention_0/Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_0/Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
DMGAttention_0/Reshape_23/shapePack"DMGAttention_0/unstack_14:output:0"DMGAttention_0/unstack_14:output:1"DMGAttention_0/unstack_15:output:0*DMGAttention_0/Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_0/Reshape_23Reshape!DMGAttention_0/MatMul_7:product:0(DMGAttention_0/Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
'DMGAttention_0/BiasAdd_1/ReadVariableOpReadVariableOp0dmgattention_0_biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0Ф
DMGAttention_0/BiasAdd_1BiasAdd"DMGAttention_0/Reshape_23:output:0/DMGAttention_0/BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Й
DMGAttention_0/stackPackDMGAttention_0/BiasAdd:output:0!DMGAttention_0/BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ g
%DMGAttention_0/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : Ж
DMGAttention_0/MeanMeanDMGAttention_0/stack:output:0.DMGAttention_0/Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"DMGAttention_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            y
$DMGAttention_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           y
$DMGAttention_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         л
DMGAttention_0/strided_sliceStridedSliceDMGAttention_0/Mean:output:0+DMGAttention_0/strided_slice/stack:output:0-DMGAttention_0/strided_slice/stack_1:output:0-DMGAttention_0/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_mask
DMGAttention_0/EluElu%DMGAttention_0/strided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ d
DMGAttention_1/ShapeShape DMGAttention_0/Elu:activations:0*
T0*
_output_shapes
:o
DMGAttention_1/unstackUnpackDMGAttention_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_1/ReadVariableOpReadVariableOp.dmgattention_1_shape_1_readvariableop_resource*
_output_shapes

:  *
dtype0g
DMGAttention_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"        q
DMGAttention_1/unstack_1UnpackDMGAttention_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numm
DMGAttention_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
DMGAttention_1/ReshapeReshape DMGAttention_0/Elu:activations:0%DMGAttention_1/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'DMGAttention_1/transpose/ReadVariableOpReadVariableOp.dmgattention_1_shape_1_readvariableop_resource*
_output_shapes

:  *
dtype0n
DMGAttention_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ї
DMGAttention_1/transpose	Transpose/DMGAttention_1/transpose/ReadVariableOp:value:0&DMGAttention_1/transpose/perm:output:0*
T0*
_output_shapes

:  o
DMGAttention_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_1ReshapeDMGAttention_1/transpose:y:0'DMGAttention_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:  
DMGAttention_1/MatMulMatMulDMGAttention_1/Reshape:output:0!DMGAttention_1/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ b
 DMGAttention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : С
DMGAttention_1/Reshape_2/shapePackDMGAttention_1/unstack:output:0DMGAttention_1/unstack:output:1)DMGAttention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Ќ
DMGAttention_1/Reshape_2ReshapeDMGAttention_1/MatMul:product:0'DMGAttention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ g
DMGAttention_1/Shape_2Shape!DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_2UnpackDMGAttention_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_3/ReadVariableOpReadVariableOp.dmgattention_1_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_1/unstack_3UnpackDMGAttention_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_1/Reshape_3Reshape!DMGAttention_1/Reshape_2:output:0'DMGAttention_1/Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_1/ReadVariableOpReadVariableOp.dmgattention_1_shape_3_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_1	Transpose1DMGAttention_1/transpose_1/ReadVariableOp:value:0(DMGAttention_1/transpose_1/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_4ReshapeDMGAttention_1/transpose_1:y:0'DMGAttention_1/Reshape_4/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_1MatMul!DMGAttention_1/Reshape_3:output:0!DMGAttention_1/Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_1/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_1/Reshape_5/shapePack!DMGAttention_1/unstack_2:output:0!DMGAttention_1/unstack_2:output:1)DMGAttention_1/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_1/Reshape_5Reshape!DMGAttention_1/MatMul_1:product:0'DMGAttention_1/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџg
DMGAttention_1/Shape_4Shape!DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_4UnpackDMGAttention_1/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_5/ReadVariableOpReadVariableOp.dmgattention_1_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0g
DMGAttention_1/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       q
DMGAttention_1/unstack_5UnpackDMGAttention_1/Shape_5:output:0*
T0*
_output_shapes
: : *	
numo
DMGAttention_1/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ё
DMGAttention_1/Reshape_6Reshape!DMGAttention_1/Reshape_2:output:0'DMGAttention_1/Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_2/ReadVariableOpReadVariableOp.dmgattention_1_shape_5_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_2	Transpose1DMGAttention_1/transpose_2/ReadVariableOp:value:0(DMGAttention_1/transpose_2/perm:output:0*
T0*
_output_shapes

: o
DMGAttention_1/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_7ReshapeDMGAttention_1/transpose_2:y:0'DMGAttention_1/Reshape_7/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_2MatMul!DMGAttention_1/Reshape_6:output:0!DMGAttention_1/Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
 DMGAttention_1/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Х
DMGAttention_1/Reshape_8/shapePack!DMGAttention_1/unstack_4:output:0!DMGAttention_1/unstack_4:output:1)DMGAttention_1/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:Ў
DMGAttention_1/Reshape_8Reshape!DMGAttention_1/MatMul_2:product:0'DMGAttention_1/Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Г
DMGAttention_1/transpose_3	Transpose!DMGAttention_1/Reshape_8:output:0(DMGAttention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџІ
DMGAttention_1/addAddV2!DMGAttention_1/Reshape_5:output:0DMGAttention_1/transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
DMGAttention_1/LeakyRelu	LeakyReluDMGAttention_1/add:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_1/subSubDMGAttention_1/sub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџY
DMGAttention_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_1/mulMulDMGAttention_1/mul/x:output:0DMGAttention_1/sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
DMGAttention_1/add_1AddV2&DMGAttention_1/LeakyRelu:activations:0DMGAttention_1/mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_1/SoftmaxSoftmaxDMGAttention_1/add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?є
;DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/MulMul DMGAttention_1/Softmax:softmax:0FDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/ShapeShape DMGAttention_1/Softmax:softmax:0*
T0*
_output_shapes
:
TDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/random_uniform/RandomUniformRandomUniformFDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0
FDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ь
DDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/GreaterEqualGreaterEqual]DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/random_uniform/RandomUniform:output:0ODMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџх
<DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/CastCastHDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/Mul_1Mul?DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/Mul:z:0@DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
;DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?ш
9DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/MulMul!DMGAttention_1/Reshape_2:output:0DDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
;DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/ShapeShape!DMGAttention_1/Reshape_2:output:0*
T0*
_output_shapes
:ѕ
RDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/random_uniform/RandomUniformRandomUniformDDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0
DDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Н
BDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/GreaterEqualGreaterEqual[DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/random_uniform/RandomUniform:output:0MDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ и
:DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/CastCastFDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
;DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/Mul_1Mul=DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/Mul:z:0>DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGAttention_1/Shape_6ShapeADMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_6UnpackDMGAttention_1/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num
DMGAttention_1/Shape_7Shape?DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_7UnpackDMGAttention_1/Shape_7:output:0*
T0*
_output_shapes
: : : *	
numk
 DMGAttention_1/Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЂ
DMGAttention_1/Reshape_9/shapePack)DMGAttention_1/Reshape_9/shape/0:output:0!DMGAttention_1/unstack_6:output:2*
N*
T0*
_output_shapes
:Ъ
DMGAttention_1/Reshape_9ReshapeADMGAttention_1/DMGAttention_1_Attention_Dropout/dropout/Mul_1:z:0'DMGAttention_1/Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          б
DMGAttention_1/transpose_4	Transpose?DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout/Mul_1:z:0(DMGAttention_1/transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_1/Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЄ
DMGAttention_1/Reshape_10/shapePack!DMGAttention_1/unstack_7:output:1*DMGAttention_1/Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_1/Reshape_10ReshapeDMGAttention_1/transpose_4:y:0(DMGAttention_1/Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЃ
DMGAttention_1/MatMul_3MatMul!DMGAttention_1/Reshape_9:output:0"DMGAttention_1/Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_1/Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : ъ
DMGAttention_1/Reshape_11/shapePack!DMGAttention_1/unstack_6:output:0!DMGAttention_1/unstack_6:output:1!DMGAttention_1/unstack_7:output:0*DMGAttention_1/Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_1/Reshape_11Reshape!DMGAttention_1/MatMul_3:product:0(DMGAttention_1/Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
%DMGAttention_1/BiasAdd/ReadVariableOpReadVariableOp.dmgattention_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Р
DMGAttention_1/BiasAddBiasAdd"DMGAttention_1/Reshape_11:output:0-DMGAttention_1/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ f
DMGAttention_1/Shape_8Shape DMGAttention_0/Elu:activations:0*
T0*
_output_shapes
:s
DMGAttention_1/unstack_8UnpackDMGAttention_1/Shape_8:output:0*
T0*
_output_shapes
: : : *	
num
%DMGAttention_1/Shape_9/ReadVariableOpReadVariableOp.dmgattention_1_shape_9_readvariableop_resource*
_output_shapes

:  *
dtype0g
DMGAttention_1/Shape_9Const*
_output_shapes
:*
dtype0*
valueB"        q
DMGAttention_1/unstack_9UnpackDMGAttention_1/Shape_9:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_1/Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Ђ
DMGAttention_1/Reshape_12Reshape DMGAttention_0/Elu:activations:0(DMGAttention_1/Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_5/ReadVariableOpReadVariableOp.dmgattention_1_shape_9_readvariableop_resource*
_output_shapes

:  *
dtype0p
DMGAttention_1/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_5	Transpose1DMGAttention_1/transpose_5/ReadVariableOp:value:0(DMGAttention_1/transpose_5/perm:output:0*
T0*
_output_shapes

:  p
DMGAttention_1/Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_13ReshapeDMGAttention_1/transpose_5:y:0(DMGAttention_1/Reshape_13/shape:output:0*
T0*
_output_shapes

:  
DMGAttention_1/MatMul_4MatMul"DMGAttention_1/Reshape_12:output:0"DMGAttention_1/Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c
!DMGAttention_1/Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : Ч
DMGAttention_1/Reshape_14/shapePack!DMGAttention_1/unstack_8:output:0!DMGAttention_1/unstack_8:output:1*DMGAttention_1/Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_1/Reshape_14Reshape!DMGAttention_1/MatMul_4:product:0(DMGAttention_1/Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
DMGAttention_1/Shape_10Shape"DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_10Unpack DMGAttention_1/Shape_10:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_1/Shape_11/ReadVariableOpReadVariableOp/dmgattention_1_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_1/Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_1/unstack_11Unpack DMGAttention_1/Shape_11:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_1/Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_1/Reshape_15Reshape"DMGAttention_1/Reshape_14:output:0(DMGAttention_1/Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_6/ReadVariableOpReadVariableOp/dmgattention_1_shape_11_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_6	Transpose1DMGAttention_1/transpose_6/ReadVariableOp:value:0(DMGAttention_1/transpose_6/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_1/Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_16ReshapeDMGAttention_1/transpose_6:y:0(DMGAttention_1/Reshape_16/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_5MatMul"DMGAttention_1/Reshape_15:output:0"DMGAttention_1/Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_1/Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_1/Reshape_17/shapePack"DMGAttention_1/unstack_10:output:0"DMGAttention_1/unstack_10:output:1*DMGAttention_1/Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_1/Reshape_17Reshape!DMGAttention_1/MatMul_5:product:0(DMGAttention_1/Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџi
DMGAttention_1/Shape_12Shape"DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_12Unpack DMGAttention_1/Shape_12:output:0*
T0*
_output_shapes
: : : *	
num
&DMGAttention_1/Shape_13/ReadVariableOpReadVariableOp/dmgattention_1_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0h
DMGAttention_1/Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       s
DMGAttention_1/unstack_13Unpack DMGAttention_1/Shape_13:output:0*
T0*
_output_shapes
: : *	
nump
DMGAttention_1/Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    Є
DMGAttention_1/Reshape_18Reshape"DMGAttention_1/Reshape_14:output:0(DMGAttention_1/Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
)DMGAttention_1/transpose_7/ReadVariableOpReadVariableOp/dmgattention_1_shape_13_readvariableop_resource*
_output_shapes

: *
dtype0p
DMGAttention_1/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       ­
DMGAttention_1/transpose_7	Transpose1DMGAttention_1/transpose_7/ReadVariableOp:value:0(DMGAttention_1/transpose_7/perm:output:0*
T0*
_output_shapes

: p
DMGAttention_1/Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
DMGAttention_1/Reshape_19ReshapeDMGAttention_1/transpose_7:y:0(DMGAttention_1/Reshape_19/shape:output:0*
T0*
_output_shapes

: 
DMGAttention_1/MatMul_6MatMul"DMGAttention_1/Reshape_18:output:0"DMGAttention_1/Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!DMGAttention_1/Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Щ
DMGAttention_1/Reshape_20/shapePack"DMGAttention_1/unstack_12:output:0"DMGAttention_1/unstack_12:output:1*DMGAttention_1/Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:А
DMGAttention_1/Reshape_20Reshape!DMGAttention_1/MatMul_6:product:0(DMGAttention_1/Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          Д
DMGAttention_1/transpose_8	Transpose"DMGAttention_1/Reshape_20:output:0(DMGAttention_1/transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЉ
DMGAttention_1/add_2AddV2"DMGAttention_1/Reshape_17:output:0DMGAttention_1/transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_1/LeakyRelu_1	LeakyReluDMGAttention_1/add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_1/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
DMGAttention_1/sub_1SubDMGAttention_1/sub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ[
DMGAttention_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ља
DMGAttention_1/mul_1MulDMGAttention_1/mul_1/x:output:0DMGAttention_1/sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
DMGAttention_1/add_3AddV2(DMGAttention_1/LeakyRelu_1:activations:0DMGAttention_1/mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGAttention_1/Softmax_1SoftmaxDMGAttention_1/add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
?DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?њ
=DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/MulMul"DMGAttention_1/Softmax_1:softmax:0HDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
?DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/ShapeShape"DMGAttention_1/Softmax_1:softmax:0*
T0*
_output_shapes
:
VDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/random_uniform/RandomUniformRandomUniformHDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0
HDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<в
FDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqualGreaterEqual_DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/random_uniform/RandomUniform:output:0QDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџщ
>DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/CastCastJDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
?DMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/Mul_1MulADMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/Mul:z:0BDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
=DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?э
;DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/MulMul"DMGAttention_1/Reshape_14:output:0FDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
=DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/ShapeShape"DMGAttention_1/Reshape_14:output:0*
T0*
_output_shapes
:љ
TDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/random_uniform/RandomUniformRandomUniformFDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0
FDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<У
DDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqualGreaterEqual]DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/random_uniform/RandomUniform:output:0ODMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ м
<DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/CastCastHDMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
=DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/Mul_1Mul?DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/Mul:z:0@DMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGAttention_1/Shape_14ShapeCDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_14Unpack DMGAttention_1/Shape_14:output:0*
T0*
_output_shapes
: : : *	
num
DMGAttention_1/Shape_15ShapeADMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:u
DMGAttention_1/unstack_15Unpack DMGAttention_1/Shape_15:output:0*
T0*
_output_shapes
: : : *	
numl
!DMGAttention_1/Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_1/Reshape_21/shapePack*DMGAttention_1/Reshape_21/shape/0:output:0"DMGAttention_1/unstack_14:output:2*
N*
T0*
_output_shapes
:Ю
DMGAttention_1/Reshape_21ReshapeCDMGAttention_1/DMGAttention_1_Attention_Dropout/dropout_1/Mul_1:z:0(DMGAttention_1/Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџt
DMGAttention_1/transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
DMGAttention_1/transpose_9	TransposeADMGAttention_1/DMGAttention_1_Feature_Dropout/dropout_1/Mul_1:z:0(DMGAttention_1/transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
!DMGAttention_1/Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
DMGAttention_1/Reshape_22/shapePack"DMGAttention_1/unstack_15:output:1*DMGAttention_1/Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:Љ
DMGAttention_1/Reshape_22ReshapeDMGAttention_1/transpose_9:y:0(DMGAttention_1/Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЄ
DMGAttention_1/MatMul_7MatMul"DMGAttention_1/Reshape_21:output:0"DMGAttention_1/Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџc
!DMGAttention_1/Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : э
DMGAttention_1/Reshape_23/shapePack"DMGAttention_1/unstack_14:output:0"DMGAttention_1/unstack_14:output:1"DMGAttention_1/unstack_15:output:0*DMGAttention_1/Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:Н
DMGAttention_1/Reshape_23Reshape!DMGAttention_1/MatMul_7:product:0(DMGAttention_1/Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
'DMGAttention_1/BiasAdd_1/ReadVariableOpReadVariableOp0dmgattention_1_biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0Ф
DMGAttention_1/BiasAdd_1BiasAdd"DMGAttention_1/Reshape_23:output:0/DMGAttention_1/BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Й
DMGAttention_1/stackPackDMGAttention_1/BiasAdd:output:0!DMGAttention_1/BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ g
%DMGAttention_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : Ж
DMGAttention_1/MeanMeanDMGAttention_1/stack:output:0.DMGAttention_1/Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ w
"DMGAttention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            y
$DMGAttention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           y
$DMGAttention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         л
DMGAttention_1/strided_sliceStridedSliceDMGAttention_1/Mean:output:0+DMGAttention_1/strided_slice/stack:output:0-DMGAttention_1/strided_slice/stack_1:output:0-DMGAttention_1/strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_mask
DMGAttention_1/EluElu%DMGAttention_1/strided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ d
"DMGReduce_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
DMGReduce_1/MeanMean DMGAttention_1/Elu:activations:0+DMGReduce_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
&DMDense_Hidden_0/MatMul/ReadVariableOpReadVariableOp/dmdense_hidden_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
DMDense_Hidden_0/MatMulMatMulDMGReduce_1/Mean:output:0.DMDense_Hidden_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'DMDense_Hidden_0/BiasAdd/ReadVariableOpReadVariableOp0dmdense_hidden_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
DMDense_Hidden_0/BiasAddBiasAdd!DMDense_Hidden_0/MatMul:product:0/DMDense_Hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
DMDense_Hidden_0/EluElu!DMDense_Hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
&DMDense_Hidden_1/MatMul/ReadVariableOpReadVariableOp/dmdense_hidden_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ї
DMDense_Hidden_1/MatMulMatMul"DMDense_Hidden_0/Elu:activations:0.DMDense_Hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'DMDense_Hidden_1/BiasAdd/ReadVariableOpReadVariableOp0dmdense_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
DMDense_Hidden_1/BiasAddBiasAdd!DMDense_Hidden_1/MatMul:product:0/DMDense_Hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
DMDense_Hidden_1/EluElu!DMDense_Hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!DMDense_OUT/MatMul/ReadVariableOpReadVariableOp*dmdense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
DMDense_OUT/MatMulMatMul"DMDense_Hidden_1/Elu:activations:0)DMDense_OUT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
"DMDense_OUT/BiasAdd/ReadVariableOpReadVariableOp+dmdense_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
DMDense_OUT/BiasAddBiasAddDMDense_OUT/MatMul:product:0*DMDense_OUT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџk
IdentityIdentityDMDense_OUT/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџх
NoOpNoOp(^DMDense_Hidden_0/BiasAdd/ReadVariableOp'^DMDense_Hidden_0/MatMul/ReadVariableOp(^DMDense_Hidden_1/BiasAdd/ReadVariableOp'^DMDense_Hidden_1/MatMul/ReadVariableOp#^DMDense_OUT/BiasAdd/ReadVariableOp"^DMDense_OUT/MatMul/ReadVariableOp&^DMGAttention_0/BiasAdd/ReadVariableOp(^DMGAttention_0/BiasAdd_1/ReadVariableOp(^DMGAttention_0/transpose/ReadVariableOp*^DMGAttention_0/transpose_1/ReadVariableOp*^DMGAttention_0/transpose_2/ReadVariableOp*^DMGAttention_0/transpose_5/ReadVariableOp*^DMGAttention_0/transpose_6/ReadVariableOp*^DMGAttention_0/transpose_7/ReadVariableOp&^DMGAttention_1/BiasAdd/ReadVariableOp(^DMGAttention_1/BiasAdd_1/ReadVariableOp(^DMGAttention_1/transpose/ReadVariableOp*^DMGAttention_1/transpose_1/ReadVariableOp*^DMGAttention_1/transpose_2/ReadVariableOp*^DMGAttention_1/transpose_5/ReadVariableOp*^DMGAttention_1/transpose_6/ReadVariableOp*^DMGAttention_1/transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2R
'DMDense_Hidden_0/BiasAdd/ReadVariableOp'DMDense_Hidden_0/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_0/MatMul/ReadVariableOp&DMDense_Hidden_0/MatMul/ReadVariableOp2R
'DMDense_Hidden_1/BiasAdd/ReadVariableOp'DMDense_Hidden_1/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_1/MatMul/ReadVariableOp&DMDense_Hidden_1/MatMul/ReadVariableOp2H
"DMDense_OUT/BiasAdd/ReadVariableOp"DMDense_OUT/BiasAdd/ReadVariableOp2F
!DMDense_OUT/MatMul/ReadVariableOp!DMDense_OUT/MatMul/ReadVariableOp2N
%DMGAttention_0/BiasAdd/ReadVariableOp%DMGAttention_0/BiasAdd/ReadVariableOp2R
'DMGAttention_0/BiasAdd_1/ReadVariableOp'DMGAttention_0/BiasAdd_1/ReadVariableOp2R
'DMGAttention_0/transpose/ReadVariableOp'DMGAttention_0/transpose/ReadVariableOp2V
)DMGAttention_0/transpose_1/ReadVariableOp)DMGAttention_0/transpose_1/ReadVariableOp2V
)DMGAttention_0/transpose_2/ReadVariableOp)DMGAttention_0/transpose_2/ReadVariableOp2V
)DMGAttention_0/transpose_5/ReadVariableOp)DMGAttention_0/transpose_5/ReadVariableOp2V
)DMGAttention_0/transpose_6/ReadVariableOp)DMGAttention_0/transpose_6/ReadVariableOp2V
)DMGAttention_0/transpose_7/ReadVariableOp)DMGAttention_0/transpose_7/ReadVariableOp2N
%DMGAttention_1/BiasAdd/ReadVariableOp%DMGAttention_1/BiasAdd/ReadVariableOp2R
'DMGAttention_1/BiasAdd_1/ReadVariableOp'DMGAttention_1/BiasAdd_1/ReadVariableOp2R
'DMGAttention_1/transpose/ReadVariableOp'DMGAttention_1/transpose/ReadVariableOp2V
)DMGAttention_1/transpose_1/ReadVariableOp)DMGAttention_1/transpose_1/ReadVariableOp2V
)DMGAttention_1/transpose_2/ReadVariableOp)DMGAttention_1/transpose_2/ReadVariableOp2V
)DMGAttention_1/transpose_5/ReadVariableOp)DMGAttention_1/transpose_5/ReadVariableOp2V
)DMGAttention_1/transpose_6/ReadVariableOp)DMGAttention_1/transpose_6/ReadVariableOp2V
)DMGAttention_1/transpose_7/ReadVariableOp)DMGAttention_1/transpose_7/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
І
ш
1__inference_DMGAttention_1_layer_call_fn_51635930
inputs_0
inputs_1
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6: 
identity

identity_1ЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633334|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1

ш
&__inference_signature_wrapper_51634432
adjacency_matrix
feature_matrix
unknown:( 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:( 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallfeature_matrixadjacency_matrixunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8 *,
f'R%
#__inference__wrapped_model_51632906o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ(: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:o k
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
*
_user_specified_nameAdjacency_Matrix:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
(
_user_specified_nameFeature_Matrix
І
ш
1__inference_DMGAttention_0_layer_call_fn_51635474
inputs_0
inputs_1
unknown:( 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:( 
	unknown_4: 
	unknown_5: 
	unknown_6: 
identity

identity_1ЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51634038|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
сЩ
§
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51634038

inputs
inputs_11
shape_1_readvariableop_resource:( 1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:( 2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:( `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:( h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_0_Attention_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Ч
,DMGAttention_0_Attention_Dropout/dropout/MulMulSoftmax:softmax:07DMGAttention_0_Attention_Dropout/dropout/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџo
.DMGAttention_0_Attention_Dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:ф
EDMGAttention_0_Attention_Dropout/dropout/random_uniform/RandomUniformRandomUniform7DMGAttention_0_Attention_Dropout/dropout/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0|
7DMGAttention_0_Attention_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_0_Attention_Dropout/dropout/GreaterEqualGreaterEqualNDMGAttention_0_Attention_Dropout/dropout/random_uniform/RandomUniform:output:0@DMGAttention_0_Attention_Dropout/dropout/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЧ
-DMGAttention_0_Attention_Dropout/dropout/CastCast9DMGAttention_0_Attention_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџт
.DMGAttention_0_Attention_Dropout/dropout/Mul_1Mul0DMGAttention_0_Attention_Dropout/dropout/Mul:z:01DMGAttention_0_Attention_Dropout/dropout/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџq
,DMGAttention_0_Feature_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Л
*DMGAttention_0_Feature_Dropout/dropout/MulMulReshape_2:output:05DMGAttention_0_Feature_Dropout/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
,DMGAttention_0_Feature_Dropout/dropout/ShapeShapeReshape_2:output:0*
T0*
_output_shapes
:з
CDMGAttention_0_Feature_Dropout/dropout/random_uniform/RandomUniformRandomUniform5DMGAttention_0_Feature_Dropout/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0z
5DMGAttention_0_Feature_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
3DMGAttention_0_Feature_Dropout/dropout/GreaterEqualGreaterEqualLDMGAttention_0_Feature_Dropout/dropout/random_uniform/RandomUniform:output:0>DMGAttention_0_Feature_Dropout/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ К
+DMGAttention_0_Feature_Dropout/dropout/CastCast7DMGAttention_0_Feature_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ г
,DMGAttention_0_Feature_Dropout/dropout/Mul_1Mul.DMGAttention_0_Feature_Dropout/dropout/Mul:z:0/DMGAttention_0_Feature_Dropout/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ i
Shape_6Shape2DMGAttention_0_Attention_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numg
Shape_7Shape0DMGAttention_0_Feature_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:
	Reshape_9Reshape2DMGAttention_0_Attention_Dropout/dropout/Mul_1:z:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Є
transpose_4	Transpose0DMGAttention_0_Feature_Dropout/dropout/Mul_1:z:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ =
Shape_8Shapeinputs*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   j

Reshape_12ReshapeinputsReshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:( a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:( n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
0DMGAttention_0_Attention_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Э
.DMGAttention_0_Attention_Dropout/dropout_1/MulMulSoftmax_1:softmax:09DMGAttention_0_Attention_Dropout/dropout_1/Const:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
0DMGAttention_0_Attention_Dropout/dropout_1/ShapeShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:ш
GDMGAttention_0_Attention_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform9DMGAttention_0_Attention_Dropout/dropout_1/Shape:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0~
9DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<Ѕ
7DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqualGreaterEqualPDMGAttention_0_Attention_Dropout/dropout_1/random_uniform/RandomUniform:output:0BDMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџЫ
/DMGAttention_0_Attention_Dropout/dropout_1/CastCast;DMGAttention_0_Attention_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџш
0DMGAttention_0_Attention_Dropout/dropout_1/Mul_1Mul2DMGAttention_0_Attention_Dropout/dropout_1/Mul:z:03DMGAttention_0_Attention_Dropout/dropout_1/Cast:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџs
.DMGAttention_0_Feature_Dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?Р
,DMGAttention_0_Feature_Dropout/dropout_1/MulMulReshape_14:output:07DMGAttention_0_Feature_Dropout/dropout_1/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ q
.DMGAttention_0_Feature_Dropout/dropout_1/ShapeShapeReshape_14:output:0*
T0*
_output_shapes
:л
EDMGAttention_0_Feature_Dropout/dropout_1/random_uniform/RandomUniformRandomUniform7DMGAttention_0_Feature_Dropout/dropout_1/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
dtype0|
7DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<
5DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqualGreaterEqualNDMGAttention_0_Feature_Dropout/dropout_1/random_uniform/RandomUniform:output:0@DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ О
-DMGAttention_0_Feature_Dropout/dropout_1/CastCast9DMGAttention_0_Feature_Dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ й
.DMGAttention_0_Feature_Dropout/dropout_1/Mul_1Mul0DMGAttention_0_Feature_Dropout/dropout_1/Mul:z:01DMGAttention_0_Feature_Dropout/dropout_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ l
Shape_14Shape4DMGAttention_0_Attention_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numj
Shape_15Shape2DMGAttention_0_Feature_Dropout/dropout_1/Mul_1:z:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:Ё

Reshape_21Reshape4DMGAttention_0_Attention_Dropout/dropout_1/Mul_1:z:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          І
transpose_9	Transpose2DMGAttention_0_Feature_Dropout/dropout_1/Mul_1:z:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј
ъ
(__inference_model_layer_call_fn_51634266
feature_matrix
adjacency_matrix
unknown:( 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:( 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallfeature_matrixadjacency_matrixunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51634169o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
(
_user_specified_nameFeature_Matrix:ok
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
*
_user_specified_nameAdjacency_Matrix
І
ш
1__inference_DMGAttention_1_layer_call_fn_51635954
inputs_0
inputs_1
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6: 
identity

identity_1ЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633757|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
ў
м
(__inference_model_layer_call_fn_51634482
inputs_0
inputs_1
unknown:( 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:( 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51633413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Д


N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51636419
input_tensor0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMulMatMulinput_tensorMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџN
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:U Q
'
_output_shapes
:џџџџџџџџџ 
&
_user_specified_nameinput_tensor

§
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633334

inputs
inputs_11
shape_1_readvariableop_resource:  1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:  2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџH
Shape_6ShapeSoftmax:softmax:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numI
Shape_7ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:|
	Reshape_9ReshapeSoftmax:softmax:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_4	TransposeReshape_2:output:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ =
Shape_8Shapeinputs*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    j

Reshape_12ReshapeinputsReshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:  a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:  n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџK
Shape_14ShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numK
Shape_15ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:

Reshape_21ReshapeSoftmax_1:softmax:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeReshape_14:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
я+
ц	
C__inference_model_layer_call_and_return_conditional_losses_51634323
feature_matrix
adjacency_matrix)
dmgattention_0_51634270:( )
dmgattention_0_51634272: )
dmgattention_0_51634274: %
dmgattention_0_51634276: )
dmgattention_0_51634278:( )
dmgattention_0_51634280: )
dmgattention_0_51634282: %
dmgattention_0_51634284: )
dmgattention_1_51634288:  )
dmgattention_1_51634290: )
dmgattention_1_51634292: %
dmgattention_1_51634294: )
dmgattention_1_51634296:  )
dmgattention_1_51634298: )
dmgattention_1_51634300: %
dmgattention_1_51634302: +
dmdense_hidden_0_51634307: '
dmdense_hidden_0_51634309:+
dmdense_hidden_1_51634312:'
dmdense_hidden_1_51634314:&
dmdense_out_51634317:"
dmdense_out_51634319:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂ&DMGAttention_0/StatefulPartitionedCallЂ&DMGAttention_1/StatefulPartitionedCall
&DMGAttention_0/StatefulPartitionedCallStatefulPartitionedCallfeature_matrixadjacency_matrixdmgattention_0_51634270dmgattention_0_51634272dmgattention_0_51634274dmgattention_0_51634276dmgattention_0_51634278dmgattention_0_51634280dmgattention_0_51634282dmgattention_0_51634284*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51633115Ц
&DMGAttention_1/StatefulPartitionedCallStatefulPartitionedCall/DMGAttention_0/StatefulPartitionedCall:output:0/DMGAttention_0/StatefulPartitionedCall:output:1dmgattention_1_51634288dmgattention_1_51634290dmgattention_1_51634292dmgattention_1_51634294dmgattention_1_51634296dmgattention_1_51634298dmgattention_1_51634300dmgattention_1_51634302*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51633334
DMGReduce_1/PartitionedCallPartitionedCall/DMGAttention_1/StatefulPartitionedCall:output:0/DMGAttention_1/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51633360И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51634307dmdense_hidden_0_51634309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51633373Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51634312dmdense_hidden_1_51634314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *W
fRRP
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51633390Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51634317dmdense_out_51634319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *R
fMRK
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51633406{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall'^DMGAttention_0/StatefulPartitionedCall'^DMGAttention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2P
&DMGAttention_0/StatefulPartitionedCall&DMGAttention_0/StatefulPartitionedCall2P
&DMGAttention_1/StatefulPartitionedCall&DMGAttention_1/StatefulPartitionedCall:d `
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
(
_user_specified_nameFeature_Matrix:ok
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
*
_user_specified_nameAdjacency_Matrix
Ј
џ
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636154
inputs_0
inputs_11
shape_1_readvariableop_resource:  1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:  2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    f
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:  *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:  `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:  h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџH
Shape_6ShapeSoftmax:softmax:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numI
Shape_7ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:|
	Reshape_9ReshapeSoftmax:softmax:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_4	TransposeReshape_2:output:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ?
Shape_8Shapeinputs_0*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"        S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    l

Reshape_12Reshapeinputs_0Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:  *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:  a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:  n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџK
Shape_14ShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numK
Shape_15ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:

Reshape_21ReshapeSoftmax_1:softmax:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeReshape_14:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
Ј
џ
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635674
inputs_0
inputs_11
shape_1_readvariableop_resource:( 1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:( 2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   f
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:( `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:( h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџH
Shape_6ShapeSoftmax:softmax:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numI
Shape_7ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:|
	Reshape_9ReshapeSoftmax:softmax:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_4	TransposeReshape_2:output:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ?
Shape_8Shapeinputs_0*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   l

Reshape_12Reshapeinputs_0Reshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:( a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:( n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџK
Shape_14ShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numK
Shape_15ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:

Reshape_21ReshapeSoftmax_1:softmax:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeReshape_14:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1
ў
м
(__inference_model_layer_call_fn_51634532
inputs_0
inputs_1
unknown:( 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:( 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51634169o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1

§
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51633115

inputs
inputs_11
shape_1_readvariableop_resource:( 1
shape_3_readvariableop_resource: 1
shape_5_readvariableop_resource: -
biasadd_readvariableop_resource: 1
shape_9_readvariableop_resource:( 2
 shape_11_readvariableop_resource: 2
 shape_13_readvariableop_resource: /
!biasadd_1_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂBiasAdd_1/ReadVariableOpЂtranspose/ReadVariableOpЂtranspose_1/ReadVariableOpЂtranspose_2/ReadVariableOpЂtranspose_5/ReadVariableOpЂtranspose_6/ReadVariableOpЂtranspose_7/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:( *
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:( `
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:( h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ I
Shape_2ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_3ReshapeReshape_2:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_1/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

: `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_4Reshapetranspose_1:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

: l
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0unstack_2:output:1Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџI
Shape_4ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"       S
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    t
	Reshape_6ReshapeReshape_2:output:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ z
transpose_2/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes

: `
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
	Reshape_7Reshapetranspose_2:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

: l
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_8/shapePackunstack_4:output:0unstack_4:output:1Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_3	TransposeReshape_8:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџy
addAddV2Reshape_5:output:0transpose_3:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ^
	LeakyRelu	LeakyReluadd:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?l
subSubsub/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаk
mulMulmul/x:output:0sub:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџx
add_1AddV2LeakyRelu:activations:0mul:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџe
SoftmaxSoftmax	add_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџH
Shape_6ShapeSoftmax:softmax:0*
T0*
_output_shapes
:U
	unstack_6UnpackShape_6:output:0*
T0*
_output_shapes
: : : *	
numI
Shape_7ShapeReshape_2:output:0*
T0*
_output_shapes
:U
	unstack_7UnpackShape_7:output:0*
T0*
_output_shapes
: : : *	
num\
Reshape_9/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
Reshape_9/shapePackReshape_9/shape/0:output:0unstack_6:output:2*
N*
T0*
_output_shapes
:|
	Reshape_9ReshapeSoftmax:softmax:0Reshape_9/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_4	TransposeReshape_2:output:0transpose_4/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_10/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџw
Reshape_10/shapePackunstack_7:output:1Reshape_10/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_10Reshapetranspose_4:y:0Reshape_10/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџv
MatMul_3MatMulReshape_9:output:0Reshape_10:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_11/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_11/shapePackunstack_6:output:0unstack_6:output:1unstack_7:output:0Reshape_11/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_11ReshapeMatMul_3:product:0Reshape_11/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddReshape_11:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ =
Shape_8Shapeinputs*
T0*
_output_shapes
:U
	unstack_8UnpackShape_8:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_9/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0X
Shape_9Const*
_output_shapes
:*
dtype0*
valueB"(       S
	unstack_9UnpackShape_9:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ(   j

Reshape_12ReshapeinputsReshape_12/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(z
transpose_5/ReadVariableOpReadVariableOpshape_9_readvariableop_resource*
_output_shapes

:( *
dtype0a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_5	Transpose"transpose_5/ReadVariableOp:value:0transpose_5/perm:output:0*
T0*
_output_shapes

:( a
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   џџџџj

Reshape_13Reshapetranspose_5:y:0Reshape_13/shape:output:0*
T0*
_output_shapes

:( n
MatMul_4MatMulReshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:џџџџџџџџџ T
Reshape_14/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 
Reshape_14/shapePackunstack_8:output:0unstack_8:output:1Reshape_14/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_14ReshapeMatMul_4:product:0Reshape_14/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ K
Shape_10ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_10UnpackShape_10:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_11/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_11Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_11UnpackShape_11:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_15/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_15ReshapeReshape_14:output:0Reshape_15/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_6/ReadVariableOpReadVariableOp shape_11_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_6	Transpose"transpose_6/ReadVariableOp:value:0transpose_6/perm:output:0*
T0*
_output_shapes

: a
Reshape_16/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_16Reshapetranspose_6:y:0Reshape_16/shape:output:0*
T0*
_output_shapes

: n
MatMul_5MatMulReshape_15:output:0Reshape_16:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_17/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_17/shapePackunstack_10:output:0unstack_10:output:1Reshape_17/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_17ReshapeMatMul_5:product:0Reshape_17/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
Shape_12ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_12UnpackShape_12:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_13/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0Y
Shape_13Const*
_output_shapes
:*
dtype0*
valueB"       U

unstack_13UnpackShape_13:output:0*
T0*
_output_shapes
: : *	
numa
Reshape_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    w

Reshape_18ReshapeReshape_14:output:0Reshape_18/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ {
transpose_7/ReadVariableOpReadVariableOp shape_13_readvariableop_resource*
_output_shapes

: *
dtype0a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_7	Transpose"transpose_7/ReadVariableOp:value:0transpose_7/perm:output:0*
T0*
_output_shapes

: a
Reshape_19/shapeConst*
_output_shapes
:*
dtype0*
valueB"    џџџџj

Reshape_19Reshapetranspose_7:y:0Reshape_19/shape:output:0*
T0*
_output_shapes

: n
MatMul_6MatMulReshape_18:output:0Reshape_19:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
Reshape_20/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_20/shapePackunstack_12:output:0unstack_12:output:1Reshape_20/shape/2:output:0*
N*
T0*
_output_shapes
:

Reshape_20ReshapeMatMul_6:product:0Reshape_20/shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџe
transpose_8/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_8	TransposeReshape_20:output:0transpose_8/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ|
add_2AddV2Reshape_17:output:0transpose_8:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџb
LeakyRelu_1	LeakyRelu	add_2:z:0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?p
sub_1Subsub_1/x:output:0inputs_1*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *љаq
mul_1Mulmul_1/x:output:0	sub_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
add_3AddV2LeakyRelu_1:activations:0	mul_1:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџg
	Softmax_1Softmax	add_3:z:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџK
Shape_14ShapeSoftmax_1:softmax:0*
T0*
_output_shapes
:W

unstack_14UnpackShape_14:output:0*
T0*
_output_shapes
: : : *	
numK
Shape_15ShapeReshape_14:output:0*
T0*
_output_shapes
:W

unstack_15UnpackShape_15:output:0*
T0*
_output_shapes
: : : *	
num]
Reshape_21/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_21/shapePackReshape_21/shape/0:output:0unstack_14:output:2*
N*
T0*
_output_shapes
:

Reshape_21ReshapeSoftmax_1:softmax:0Reshape_21/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџe
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_9	TransposeReshape_14:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
Reshape_22/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџx
Reshape_22/shapePackunstack_15:output:1Reshape_22/shape/1:output:0*
N*
T0*
_output_shapes
:|

Reshape_22Reshapetranspose_9:y:0Reshape_22/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџw
MatMul_7MatMulReshape_21:output:0Reshape_22:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџT
Reshape_23/shape/3Const*
_output_shapes
: *
dtype0*
value	B : Ђ
Reshape_23/shapePackunstack_14:output:0unstack_14:output:1unstack_15:output:0Reshape_23/shape/3:output:0*
N*
T0*
_output_shapes
:

Reshape_23ReshapeMatMul_7:product:0Reshape_23/shape:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ v
BiasAdd_1/ReadVariableOpReadVariableOp!biasadd_1_readvariableop_resource*
_output_shapes
: *
dtype0
	BiasAdd_1BiasAddReshape_23:output:0 BiasAdd_1/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
stackPackBiasAdd:output:0BiasAdd_1:output:0*
N*
T0*E
_output_shapes3
1:/џџџџџџџџџџџџџџџџџџџџџџџџџџџ X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
MeanMeanstack:output:0Mean/reduction_indices:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
strided_sliceStridedSliceMean:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *

begin_mask*
end_mask*
shrink_axis_maska
EluElustrided_slice:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
IdentityIdentityElu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
NoOpNoOp^BiasAdd/ReadVariableOp^BiasAdd_1/ReadVariableOp^transpose/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^transpose_5/ReadVariableOp^transpose_6/ReadVariableOp^transpose_7/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
BiasAdd_1/ReadVariableOpBiasAdd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp28
transpose_5/ReadVariableOptranspose_5/ReadVariableOp28
transpose_6/ReadVariableOptranspose_6/ReadVariableOp28
transpose_7/ReadVariableOptranspose_7/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І
ш
1__inference_DMGAttention_0_layer_call_fn_51635450
inputs_0
inputs_1
unknown:( 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3:( 
	unknown_4: 
	unknown_5: 
	unknown_6: 
identity

identity_1ЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
_read_only_resource_inputs

	*1
config_proto!

CPU

GPU (2J 8 *U
fPRN
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51633115|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/1"ПL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ў
serving_default
c
Adjacency_MatrixO
"serving_default_Adjacency_Matrix:0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
V
Feature_MatrixD
 serving_default_Feature_Matrix:0џџџџџџџџџџџџџџџџџџ(?
DMDense_OUT0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ў
О
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
К
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
w
self_attention_w
neighbor_attention_w
bias
DMGAttention_0_0_weight
*&DMGAttention_0_0_self_attention_weight
.*DMGAttention_0_0_neighbor_attention_weight
DMGAttention_0_0_bias
DMGAttention_0_1_weight
* &DMGAttention_0_1_self_attention_weight
.!*DMGAttention_0_1_neighbor_attention_weight
"DMGAttention_0_1_bias
#attention_dropout
$feature_dropout"
_tf_keras_layer
К
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+w
,self_attention_w
-neighbor_attention_w
.bias
/DMGAttention_1_0_weight
*0&DMGAttention_1_0_self_attention_weight
.1*DMGAttention_1_0_neighbor_attention_weight
2DMGAttention_1_0_bias
3DMGAttention_1_1_weight
*4&DMGAttention_1_1_self_attention_weight
.5*DMGAttention_1_1_neighbor_attention_weight
6DMGAttention_1_1_bias
7attention_dropout
8feature_dropout"
_tf_keras_layer
Ѕ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
ѓ
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
EDMDense_Hidden_0_weight

Eweight
FDMDense_Hidden_0_bias
Fbias"
_tf_keras_layer
ѓ
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
MDMDense_Hidden_1_weight

Mweight
NDMDense_Hidden_1_bias
Nbias"
_tf_keras_layer
щ
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
UDMDense_OUT_weight

Uweight
VDMDense_OUT_bias
Vbias"
_tf_keras_layer
Ц
0
1
2
3
4
 5
!6
"7
/8
09
110
211
312
413
514
615
E16
F17
M18
N19
U20
V21"
trackable_list_wrapper
Ц
0
1
2
3
4
 5
!6
"7
/8
09
110
211
312
413
514
615
E16
F17
M18
N19
U20
V21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ж
\trace_0
]trace_1
^trace_2
_trace_32ы
(__inference_model_layer_call_fn_51633460
(__inference_model_layer_call_fn_51634482
(__inference_model_layer_call_fn_51634532
(__inference_model_layer_call_fn_51634266Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 z\trace_0z]trace_1z^trace_2z_trace_3
Т
`trace_0
atrace_1
btrace_2
ctrace_32з
C__inference_model_layer_call_and_return_conditional_losses_51634947
C__inference_model_layer_call_and_return_conditional_losses_51635426
C__inference_model_layer_call_and_return_conditional_losses_51634323
C__inference_model_layer_call_and_return_conditional_losses_51634380Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 z`trace_0zatrace_1zbtrace_2zctrace_3
чBф
#__inference__wrapped_model_51632906Feature_MatrixAdjacency_Matrix"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
,
dserving_default"
signature_map
X
0
1
2
3
4
 5
!6
"7"
trackable_list_wrapper
X
0
1
2
3
4
 5
!6
"7"
trackable_list_wrapper
 "
trackable_list_wrapper
­
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
г
jtrace_0
ktrace_12
1__inference_DMGAttention_0_layer_call_fn_51635450
1__inference_DMGAttention_0_layer_call_fn_51635474Г
ЊВІ
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
annotationsЊ *
 zjtrace_0zktrace_1

ltrace_0
mtrace_12в
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635674
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635906Г
ЊВІ
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
annotationsЊ *
 zltrace_0zmtrace_1
.
0
1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
!1"
trackable_list_wrapper
.
0
"1"
trackable_list_wrapper
8:6( 2&DMGAttention_0/DMGAttention_0_0_weight
G:E 25DMGAttention_0/DMGAttention_0_0_self_attention_weight
K:I 29DMGAttention_0/DMGAttention_0_0_neighbor_attention_weight
2:0 2$DMGAttention_0/DMGAttention_0_0_bias
8:6( 2&DMGAttention_0/DMGAttention_0_1_weight
G:E 25DMGAttention_0/DMGAttention_0_1_self_attention_weight
K:I 29DMGAttention_0/DMGAttention_0_1_neighbor_attention_weight
2:0 2$DMGAttention_0/DMGAttention_0_1_bias
Ѕ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
X
/0
01
12
23
34
45
56
67"
trackable_list_wrapper
X
/0
01
12
23
34
45
56
67"
trackable_list_wrapper
 "
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
е
trace_0
trace_12
1__inference_DMGAttention_1_layer_call_fn_51635930
1__inference_DMGAttention_1_layer_call_fn_51635954Г
ЊВІ
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
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12в
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636154
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636386Г
ЊВІ
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
annotationsЊ *
 ztrace_0ztrace_1
.
/0
31"
trackable_list_wrapper
.
00
41"
trackable_list_wrapper
.
10
51"
trackable_list_wrapper
.
20
61"
trackable_list_wrapper
8:6  2&DMGAttention_1/DMGAttention_1_0_weight
G:E 25DMGAttention_1/DMGAttention_1_0_self_attention_weight
K:I 29DMGAttention_1/DMGAttention_1_0_neighbor_attention_weight
2:0 2$DMGAttention_1/DMGAttention_1_0_bias
8:6  2&DMGAttention_1/DMGAttention_1_1_weight
G:E 25DMGAttention_1/DMGAttention_1_1_self_attention_weight
K:I 29DMGAttention_1/DMGAttention_1_1_neighbor_attention_weight
2:0 2$DMGAttention_1/DMGAttention_1_1_bias
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
є
trace_02е
.__inference_DMGReduce_1_layer_call_fn_51636392Ђ
В
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
annotationsЊ *
 ztrace_0

trace_02№
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51636399Ђ
В
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
annotationsЊ *
 ztrace_0
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
џ
trace_02р
3__inference_DMDense_Hidden_0_layer_call_fn_51636408Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ћ
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51636419Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
::8 2(DMDense_Hidden_0/DMDense_Hidden_0_weight
4:22&DMDense_Hidden_0/DMDense_Hidden_0_bias
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
џ
Ђtrace_02р
3__inference_DMDense_Hidden_1_layer_call_fn_51636428Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЂtrace_0

Ѓtrace_02ћ
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51636439Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0
::82(DMDense_Hidden_1/DMDense_Hidden_1_weight
4:22&DMDense_Hidden_1/DMDense_Hidden_1_bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
њ
Љtrace_02л
.__inference_DMDense_OUT_layer_call_fn_51636448Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЉtrace_0

Њtrace_02і
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51636458Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0
0:.2DMDense_OUT/DMDense_OUT_weight
*:(2DMDense_OUT/DMDense_OUT_bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
(__inference_model_layer_call_fn_51633460Feature_MatrixAdjacency_Matrix"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_model_layer_call_fn_51634482inputs/0inputs/1"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_model_layer_call_fn_51634532inputs/0inputs/1"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
(__inference_model_layer_call_fn_51634266Feature_MatrixAdjacency_Matrix"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЁB
C__inference_model_layer_call_and_return_conditional_losses_51634947inputs/0inputs/1"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЁB
C__inference_model_layer_call_and_return_conditional_losses_51635426inputs/0inputs/1"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЏBЌ
C__inference_model_layer_call_and_return_conditional_losses_51634323Feature_MatrixAdjacency_Matrix"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЏBЌ
C__inference_model_layer_call_and_return_conditional_losses_51634380Feature_MatrixAdjacency_Matrix"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
фBс
&__inference_signature_wrapper_51634432Adjacency_MatrixFeature_Matrix"
В
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
annotationsЊ *
 
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bџ
1__inference_DMGAttention_0_layer_call_fn_51635450inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
Bџ
1__inference_DMGAttention_0_layer_call_fn_51635474inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
B
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635674inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
B
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635906inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bџ
1__inference_DMGAttention_1_layer_call_fn_51635930inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
Bџ
1__inference_DMGAttention_1_layer_call_fn_51635954inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
B
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636154inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
B
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636386inputs/0inputs/1"Г
ЊВІ
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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
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
p

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
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
юBы
.__inference_DMGReduce_1_layer_call_fn_51636392inputs/0inputs/1"Ђ
В
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
annotationsЊ *
 
B
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51636399inputs/0inputs/1"Ђ
В
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
annotationsЊ *
 
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
ѓB№
3__inference_DMDense_Hidden_0_layer_call_fn_51636408input_tensor"Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51636419input_tensor"Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
ѓB№
3__inference_DMDense_Hidden_1_layer_call_fn_51636428input_tensor"Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51636439input_tensor"Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
юBы
.__inference_DMDense_OUT_layer_call_fn_51636448input_tensor"Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51636458input_tensor"Ј
В
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
С	variables
Т	keras_api

Уtotal

Фcount"
_tf_keras_metric
c
Х	variables
Ц	keras_api

Чtotal

Шcount
Щ
_fn_kwargs"
_tf_keras_metric
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
 "
trackable_dict_wrapper
0
У0
Ф1"
trackable_list_wrapper
.
С	variables"
_generic_user_object
:  (2total
:  (2count
0
Ч0
Ш1"
trackable_list_wrapper
.
Х	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperД
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51636419bEF5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 
3__inference_DMDense_Hidden_0_layer_call_fn_51636408UEF5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ 
Њ "џџџџџџџџџД
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51636439bMN5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
3__inference_DMDense_Hidden_1_layer_call_fn_51636428UMN5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "џџџџџџџџџЏ
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51636458bUV5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_DMDense_OUT_layer_call_fn_51636448UUV5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "џџџџџџџџџЯ
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635674ў !"Ђ~
wЂt
nk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "nЂk
dЂa
*'
0/0џџџџџџџџџџџџџџџџџџ 
30
0/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
L__inference_DMGAttention_0_layer_call_and_return_conditional_losses_51635906ў !"Ђ~
wЂt
nk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "nЂk
dЂa
*'
0/0џџџџџџџџџџџџџџџџџџ 
30
0/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 І
1__inference_DMGAttention_0_layer_call_fn_51635450№ !"Ђ~
wЂt
nk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "`Ђ]
(%
0џџџџџџџџџџџџџџџџџџ 
1.
1'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
1__inference_DMGAttention_0_layer_call_fn_51635474№ !"Ђ~
wЂt
nk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "`Ђ]
(%
0џџџџџџџџџџџџџџџџџџ 
1.
1'џџџџџџџџџџџџџџџџџџџџџџџџџџџЯ
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636154ў/0123456Ђ~
wЂt
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "nЂk
dЂa
*'
0/0џџџџџџџџџџџџџџџџџџ 
30
0/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
L__inference_DMGAttention_1_layer_call_and_return_conditional_losses_51636386ў/0123456Ђ~
wЂt
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "nЂk
dЂa
*'
0/0џџџџџџџџџџџџџџџџџџ 
30
0/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 І
1__inference_DMGAttention_1_layer_call_fn_51635930№/0123456Ђ~
wЂt
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "`Ђ]
(%
0џџџџџџџџџџџџџџџџџџ 
1.
1'џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
1__inference_DMGAttention_1_layer_call_fn_51635954№/0123456Ђ~
wЂt
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "`Ђ]
(%
0џџџџџџџџџџџџџџџџџџ 
1.
1'џџџџџџџџџџџџџџџџџџџџџџџџџџџє
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51636399І}Ђz
sЂp
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ 
 Ь
.__inference_DMGReduce_1_layer_call_fn_51636392}Ђz
sЂp
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "џџџџџџџџџ 
#__inference__wrapped_model_51632906у !"/0123456EFMNUVЂ
Ђ~
|Ђy
52
Feature_Matrixџџџџџџџџџџџџџџџџџџ(
@=
Adjacency_Matrix'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "9Њ6
4
DMDense_OUT%"
DMDense_OUTџџџџџџџџџ 
C__inference_model_layer_call_and_return_conditional_losses_51634323и !"/0123456EFMNUVЂ
Ђ
|Ђy
52
Feature_Matrixџџџџџџџџџџџџџџџџџџ(
@=
Adjacency_Matrix'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
  
C__inference_model_layer_call_and_return_conditional_losses_51634380и !"/0123456EFMNUVЂ
Ђ
|Ђy
52
Feature_Matrixџџџџџџџџџџџџџџџџџџ(
@=
Adjacency_Matrix'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
C__inference_model_layer_call_and_return_conditional_losses_51634947Ш !"/0123456EFMNUVЂ
{Ђx
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
C__inference_model_layer_call_and_return_conditional_losses_51635426Ш !"/0123456EFMNUVЂ
{Ђx
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 ј
(__inference_model_layer_call_fn_51633460Ы !"/0123456EFMNUVЂ
Ђ
|Ђy
52
Feature_Matrixџџџџџџџџџџџџџџџџџџ(
@=
Adjacency_Matrix'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџј
(__inference_model_layer_call_fn_51634266Ы !"/0123456EFMNUVЂ
Ђ
|Ђy
52
Feature_Matrixџџџџџџџџџџџџџџџџџџ(
@=
Adjacency_Matrix'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџш
(__inference_model_layer_call_fn_51634482Л !"/0123456EFMNUVЂ
{Ђx
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџш
(__inference_model_layer_call_fn_51634532Л !"/0123456EFMNUVЂ
{Ђx
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџА
&__inference_signature_wrapper_51634432 !"/0123456EFMNUVЏЂЋ
Ђ 
ЃЊ
T
Adjacency_Matrix@=
Adjacency_Matrix'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
G
Feature_Matrix52
Feature_Matrixџџџџџџџџџџџџџџџџџџ("9Њ6
4
DMDense_OUT%"
DMDense_OUTџџџџџџџџџ