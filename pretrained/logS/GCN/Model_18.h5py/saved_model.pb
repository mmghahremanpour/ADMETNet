ЌЃ
Ѕ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
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
П
MatrixDiagV3
diagonal"T
k
num_rows
num_cols
padding_value"T
output"T"	
Ttype"Q
alignstring
RIGHT_LEFT:2
0
LEFT_RIGHT
RIGHT_LEFT	LEFT_LEFTRIGHT_RIGHT
S
MatrixInverse

input"T
output"T"
adjointbool( "
Ttype:	
2
A
MatrixSquareRoot

input"T
output"T"
Ttype:	
2
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12unknown8со
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

DMGCN_1/DMGCN_1_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameDMGCN_1/DMGCN_1_bias
y
(DMGCN_1/DMGCN_1_bias/Read/ReadVariableOpReadVariableOpDMGCN_1/DMGCN_1_bias*
_output_shapes
: *
dtype0

DMGCN_1/DMGCN_1_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameDMGCN_1/DMGCN_1_weight

*DMGCN_1/DMGCN_1_weight/Read/ReadVariableOpReadVariableOpDMGCN_1/DMGCN_1_weight*
_output_shapes

:  *
dtype0

DMGCN_0/DMGCN_0_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameDMGCN_0/DMGCN_0_bias
y
(DMGCN_0/DMGCN_0_bias/Read/ReadVariableOpReadVariableOpDMGCN_0/DMGCN_0_bias*
_output_shapes
: *
dtype0

DMGCN_0/DMGCN_0_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *'
shared_nameDMGCN_0/DMGCN_0_weight

*DMGCN_0/DMGCN_0_weight/Read/ReadVariableOpReadVariableOpDMGCN_0/DMGCN_0_weight*
_output_shapes

:( *
dtype0

NoOpNoOp
3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ы2
valueС2BО2 BЗ2
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
Ч
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
DMGCN_0_weight
w
DMGCN_0_bias
bias*
Ч
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
DMGCN_1_weight
w
 DMGCN_1_bias
 bias*

!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses* 
о
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-DMDense_Hidden_0_weight

-weight
.DMDense_Hidden_0_bias
.bias*
о
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5DMDense_Hidden_1_weight

5weight
6DMDense_Hidden_1_bias
6bias*
д
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=DMDense_OUT_weight

=weight
>DMDense_OUT_bias
>bias*
J
0
1
2
 3
-4
.5
56
67
=8
>9*
J
0
1
2
 3
-4
.5
56
67
=8
>9*
* 
А
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_3* 
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
* 

Lserving_default* 

0
1*

0
1*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
nh
VARIABLE_VALUEDMGCN_0/DMGCN_0_weight>layer_with_weights-0/DMGCN_0_weight/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEDMGCN_0/DMGCN_0_bias<layer_with_weights-0/DMGCN_0_bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
nh
VARIABLE_VALUEDMGCN_1/DMGCN_1_weight>layer_with_weights-1/DMGCN_1_weight/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEDMGCN_1/DMGCN_1_bias<layer_with_weights-1/DMGCN_1_bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

`trace_0* 

atrace_0* 

-0
.1*

-0
.1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 

VARIABLE_VALUE(DMDense_Hidden_0/DMDense_Hidden_0_weightGlayer_with_weights-2/DMDense_Hidden_0_weight/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&DMDense_Hidden_0/DMDense_Hidden_0_biasElayer_with_weights-2/DMDense_Hidden_0_bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 

VARIABLE_VALUE(DMDense_Hidden_1/DMDense_Hidden_1_weightGlayer_with_weights-3/DMDense_Hidden_1_weight/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE&DMDense_Hidden_1/DMDense_Hidden_1_biasElayer_with_weights-3/DMDense_Hidden_1_bias/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1*

=0
>1*
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

utrace_0* 

vtrace_0* 
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

w0
x1*
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
8
y	variables
z	keras_api
	{total
	|count*
J
}	variables
~	keras_api
	total

count

_fn_kwargs*

{0
|1*

y	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

}	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

serving_default_input_1Placeholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ(
І
serving_default_input_2Placeholder*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
dtype0*2
shape):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Є
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2DMGCN_0/DMGCN_0_weightDMGCN_0/DMGCN_0_biasDMGCN_1/DMGCN_1_weightDMGCN_1/DMGCN_1_bias(DMDense_Hidden_0/DMDense_Hidden_0_weight&DMDense_Hidden_0/DMDense_Hidden_0_bias(DMDense_Hidden_1/DMDense_Hidden_1_weight&DMDense_Hidden_1/DMDense_Hidden_1_biasDMDense_OUT/DMDense_OUT_weightDMDense_OUT/DMDense_OUT_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8 */
f*R(
&__inference_signature_wrapper_51639144
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ѕ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*DMGCN_0/DMGCN_0_weight/Read/ReadVariableOp(DMGCN_0/DMGCN_0_bias/Read/ReadVariableOp*DMGCN_1/DMGCN_1_weight/Read/ReadVariableOp(DMGCN_1/DMGCN_1_bias/Read/ReadVariableOp<DMDense_Hidden_0/DMDense_Hidden_0_weight/Read/ReadVariableOp:DMDense_Hidden_0/DMDense_Hidden_0_bias/Read/ReadVariableOp<DMDense_Hidden_1/DMDense_Hidden_1_weight/Read/ReadVariableOp:DMDense_Hidden_1/DMDense_Hidden_1_bias/Read/ReadVariableOp2DMDense_OUT/DMDense_OUT_weight/Read/ReadVariableOp0DMDense_OUT/DMDense_OUT_bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
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
!__inference__traced_save_51639538

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDMGCN_0/DMGCN_0_weightDMGCN_0/DMGCN_0_biasDMGCN_1/DMGCN_1_weightDMGCN_1/DMGCN_1_bias(DMDense_Hidden_0/DMDense_Hidden_0_weight&DMDense_Hidden_0/DMDense_Hidden_0_bias(DMDense_Hidden_1/DMDense_Hidden_1_weight&DMDense_Hidden_1/DMDense_Hidden_1_biasDMDense_OUT/DMDense_OUT_weightDMDense_OUT/DMDense_OUT_biastotal_1count_1totalcount*
Tin
2*
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
$__inference__traced_restore_51639590Вљ
й
ћ
(__inference_model_layer_call_fn_51639050
input_1
input_2
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51639001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
!
_user_specified_name	input_1:fb
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_2
З
љ
&__inference_signature_wrapper_51639144
input_1
input_2
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8 *,
f'R%
#__inference__wrapped_model_51638720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
!
_user_specified_name	input_1:fb
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_2
З

E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51639400
inputs_0
inputs_10
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџo
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџH
diag/kConst*
_output_shapes
: *
dtype0*
value	B : X
diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџX
diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџW
diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
diagMatrixDiagV3Sum:output:0diag/k:output:0diag/num_rows:output:0diag/num_cols:output:0diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
MatrixInverseMatrixInversediag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
MatrixSquareRootMatrixSquareRootMatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0
MatMulBatchMatMulV2inputs_0MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_1BatchMatMulV2MatrixSquareRoot:output:0MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ u
MatMul_2BatchMatMulV2inputs_1MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_3BatchMatMulV2MatrixSquareRoot:output:0MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddMatMul_3:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:^ Z
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
3__inference_DMDense_Hidden_0_layer_call_fn_51639422
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
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51638813o
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
й(
ў
!__inference__traced_save_51639538
file_prefix5
1savev2_dmgcn_0_dmgcn_0_weight_read_readvariableop3
/savev2_dmgcn_0_dmgcn_0_bias_read_readvariableop5
1savev2_dmgcn_1_dmgcn_1_weight_read_readvariableop3
/savev2_dmgcn_1_dmgcn_1_bias_read_readvariableopG
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
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*­
valueЃB B>layer_with_weights-0/DMGCN_0_weight/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/DMGCN_0_bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/DMGCN_1_weight/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/DMGCN_1_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-2/DMDense_Hidden_0_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/DMDense_Hidden_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-3/DMDense_Hidden_1_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-3/DMDense_Hidden_1_bias/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-4/DMDense_OUT_weight/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/DMDense_OUT_bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_dmgcn_0_dmgcn_0_weight_read_readvariableop/savev2_dmgcn_0_dmgcn_0_bias_read_readvariableop1savev2_dmgcn_1_dmgcn_1_weight_read_readvariableop/savev2_dmgcn_1_dmgcn_1_bias_read_readvariableopCsavev2_dmdense_hidden_0_dmdense_hidden_0_weight_read_readvariableopAsavev2_dmdense_hidden_0_dmdense_hidden_0_bias_read_readvariableopCsavev2_dmdense_hidden_1_dmdense_hidden_1_weight_read_readvariableopAsavev2_dmdense_hidden_1_dmdense_hidden_1_bias_read_readvariableop9savev2_dmdense_out_dmdense_out_weight_read_readvariableop7savev2_dmdense_out_dmdense_out_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
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

identity_1Identity_1:output:0*o
_input_shapes^
\: :( : :  : : :::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:( : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Џ

E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51638754

inputs
inputs_10
matmul_readvariableop_resource:( -
biasadd_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџo
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџH
diag/kConst*
_output_shapes
: *
dtype0*
value	B : X
diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџX
diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџW
diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
diagMatrixDiagV3Sum:output:0diag/k:output:0diag/num_rows:output:0diag/num_cols:output:0diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
MatrixInverseMatrixInversediag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
MatrixSquareRootMatrixSquareRootMatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype0}
MatMulBatchMatMulV2inputsMatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_1BatchMatMulV2MatrixSquareRoot:output:0MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ u
MatMul_2BatchMatMulV2inputs_1MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_3BatchMatMulV2MatrixSquareRoot:output:0MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddMatMul_3:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
З


N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51639453
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
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
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
О#
А
C__inference_model_layer_call_and_return_conditional_losses_51639083
input_1
input_2"
dmgcn_0_51639054:( 
dmgcn_0_51639056: "
dmgcn_1_51639060:  
dmgcn_1_51639062: +
dmdense_hidden_0_51639067: '
dmdense_hidden_0_51639069:+
dmdense_hidden_1_51639072:'
dmdense_hidden_1_51639074:&
dmdense_out_51639077:"
dmdense_out_51639079:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂDMGCN_0/StatefulPartitionedCallЂDMGCN_1/StatefulPartitionedCallИ
DMGCN_0/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2dmgcn_0_51639054dmgcn_0_51639056*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51638754њ
DMGCN_1/StatefulPartitionedCallStatefulPartitionedCall(DMGCN_0/StatefulPartitionedCall:output:0(DMGCN_0/StatefulPartitionedCall:output:1dmgcn_1_51639060dmgcn_1_51639062*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51638786
DMGReduce_1/PartitionedCallPartitionedCall(DMGCN_1/StatefulPartitionedCall:output:0(DMGCN_1/StatefulPartitionedCall:output:1*
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
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51638800И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51639067dmdense_hidden_0_51639069*
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
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51638813Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51639072dmdense_hidden_1_51639074*
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
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51638830Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51639077dmdense_out_51639079*
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
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51638846{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall ^DMGCN_0/StatefulPartitionedCall ^DMGCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2B
DMGCN_0/StatefulPartitionedCallDMGCN_0/StatefulPartitionedCall2B
DMGCN_1/StatefulPartitionedCallDMGCN_1/StatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
!
_user_specified_name	input_1:fb
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_2

З
*__inference_DMGCN_1_layer_call_fn_51639375
inputs_0
inputs_1
unknown:  
	unknown_0: 
identity

identity_1ЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51638786|
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
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
о	

I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51638846
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
З


N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51638830
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
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
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
Џ

E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51638786

inputs
inputs_10
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџo
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџH
diag/kConst*
_output_shapes
: *
dtype0*
value	B : X
diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџX
diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџW
diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
diagMatrixDiagV3Sum:output:0diag/k:output:0diag/num_rows:output:0diag/num_cols:output:0diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
MatrixInverseMatrixInversediag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
MatrixSquareRootMatrixSquareRootMatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0}
MatMulBatchMatMulV2inputsMatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_1BatchMatMulV2MatrixSquareRoot:output:0MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ u
MatMul_2BatchMatMulV2inputs_1MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_3BatchMatMulV2MatrixSquareRoot:output:0MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddMatMul_3:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь
І
3__inference_DMDense_Hidden_1_layer_call_fn_51639442
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
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51638830o
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
ч<
Ѕ	
$__inference__traced_restore_51639590
file_prefix9
'assignvariableop_dmgcn_0_dmgcn_0_weight:( 5
'assignvariableop_1_dmgcn_0_dmgcn_0_bias: ;
)assignvariableop_2_dmgcn_1_dmgcn_1_weight:  5
'assignvariableop_3_dmgcn_1_dmgcn_1_bias: M
;assignvariableop_4_dmdense_hidden_0_dmdense_hidden_0_weight: G
9assignvariableop_5_dmdense_hidden_0_dmdense_hidden_0_bias:M
;assignvariableop_6_dmdense_hidden_1_dmdense_hidden_1_weight:G
9assignvariableop_7_dmdense_hidden_1_dmdense_hidden_1_bias:C
1assignvariableop_8_dmdense_out_dmdense_out_weight:=
/assignvariableop_9_dmdense_out_dmdense_out_bias:%
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: 
identity_15ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*­
valueЃB B>layer_with_weights-0/DMGCN_0_weight/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-0/DMGCN_0_bias/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/DMGCN_1_weight/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/DMGCN_1_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-2/DMDense_Hidden_0_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/DMDense_Hidden_0_bias/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-3/DMDense_Hidden_1_weight/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-3/DMDense_Hidden_1_bias/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-4/DMDense_OUT_weight/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/DMDense_OUT_bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B щ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_dmgcn_0_dmgcn_0_weightIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp'assignvariableop_1_dmgcn_0_dmgcn_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp)assignvariableop_2_dmgcn_1_dmgcn_1_weightIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp'assignvariableop_3_dmgcn_1_dmgcn_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_4AssignVariableOp;assignvariableop_4_dmdense_hidden_0_dmdense_hidden_0_weightIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp9assignvariableop_5_dmdense_hidden_0_dmdense_hidden_0_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_6AssignVariableOp;assignvariableop_6_dmdense_hidden_1_dmdense_hidden_1_weightIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp9assignvariableop_7_dmdense_hidden_1_dmdense_hidden_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_8AssignVariableOp1assignvariableop_8_dmdense_out_dmdense_out_weightIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp/assignvariableop_9_dmdense_out_dmdense_out_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: №
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
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
М#
А
C__inference_model_layer_call_and_return_conditional_losses_51638853

inputs
inputs_1"
dmgcn_0_51638755:( 
dmgcn_0_51638757: "
dmgcn_1_51638787:  
dmgcn_1_51638789: +
dmdense_hidden_0_51638814: '
dmdense_hidden_0_51638816:+
dmdense_hidden_1_51638831:'
dmdense_hidden_1_51638833:&
dmdense_out_51638847:"
dmdense_out_51638849:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂDMGCN_0/StatefulPartitionedCallЂDMGCN_1/StatefulPartitionedCallИ
DMGCN_0/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dmgcn_0_51638755dmgcn_0_51638757*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51638754њ
DMGCN_1/StatefulPartitionedCallStatefulPartitionedCall(DMGCN_0/StatefulPartitionedCall:output:0(DMGCN_0/StatefulPartitionedCall:output:1dmgcn_1_51638787dmgcn_1_51638789*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51638786
DMGReduce_1/PartitionedCallPartitionedCall(DMGCN_1/StatefulPartitionedCall:output:0(DMGCN_1/StatefulPartitionedCall:output:1*
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
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51638800И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51638814dmdense_hidden_0_51638816*
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
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51638813Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51638831dmdense_hidden_1_51638833*
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
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51638830Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51638847dmdense_out_51638849*
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
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51638846{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall ^DMGCN_0/StatefulPartitionedCall ^DMGCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2B
DMGCN_0/StatefulPartitionedCallDMGCN_0/StatefulPartitionedCall2B
DMGCN_1/StatefulPartitionedCallDMGCN_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
й
ћ
(__inference_model_layer_call_fn_51638876
input_1
input_2
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51638853o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
!
_user_specified_name	input_1:fb
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_2
т
Ё
.__inference_DMDense_OUT_layer_call_fn_51639462
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
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51638846o
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
о	

I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51639472
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
З


N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51638813
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
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
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
О#
А
C__inference_model_layer_call_and_return_conditional_losses_51639116
input_1
input_2"
dmgcn_0_51639087:( 
dmgcn_0_51639089: "
dmgcn_1_51639093:  
dmgcn_1_51639095: +
dmdense_hidden_0_51639100: '
dmdense_hidden_0_51639102:+
dmdense_hidden_1_51639105:'
dmdense_hidden_1_51639107:&
dmdense_out_51639110:"
dmdense_out_51639112:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂDMGCN_0/StatefulPartitionedCallЂDMGCN_1/StatefulPartitionedCallИ
DMGCN_0/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2dmgcn_0_51639087dmgcn_0_51639089*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51638754њ
DMGCN_1/StatefulPartitionedCallStatefulPartitionedCall(DMGCN_0/StatefulPartitionedCall:output:0(DMGCN_0/StatefulPartitionedCall:output:1dmgcn_1_51639093dmgcn_1_51639095*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51638786
DMGReduce_1/PartitionedCallPartitionedCall(DMGCN_1/StatefulPartitionedCall:output:0(DMGCN_1/StatefulPartitionedCall:output:1*
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
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51638800И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51639100dmdense_hidden_0_51639102*
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
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51638813Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51639105dmdense_hidden_1_51639107*
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
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51638830Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51639110dmdense_out_51639112*
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
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51638846{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall ^DMGCN_0/StatefulPartitionedCall ^DMGCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2B
DMGCN_0/StatefulPartitionedCallDMGCN_0/StatefulPartitionedCall2B
DMGCN_1/StatefulPartitionedCallDMGCN_1/StatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
!
_user_specified_name	input_1:fb
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_2
п
§
(__inference_model_layer_call_fn_51639170
inputs_0
inputs_1
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51638853o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 22
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
ћ
s
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51638800

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

З
*__inference_DMGCN_0_layer_call_fn_51639338
inputs_0
inputs_1
unknown:( 
	unknown_0: 
identity

identity_1ЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51638754|
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
_construction_contextkEagerRuntime*`
_input_shapesO
M:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
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
п
§
(__inference_model_layer_call_fn_51639196
inputs_0
inputs_1
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_51639001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 22
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
З

E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51639363
inputs_0
inputs_10
matmul_readvariableop_resource:( -
biasadd_readvariableop_resource: 
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџo
SumSuminputs_1Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџH
diag/kConst*
_output_shapes
: *
dtype0*
value	B : X
diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџX
diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџW
diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
diagMatrixDiagV3Sum:output:0diag/k:output:0diag/num_rows:output:0diag/num_cols:output:0diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџu
MatrixInverseMatrixInversediag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
MatrixSquareRootMatrixSquareRootMatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype0
MatMulBatchMatMulV2inputs_0MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_1BatchMatMulV2MatrixSquareRoot:output:0MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ u
MatMul_2BatchMatMulV2inputs_1MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
MatMul_3BatchMatMulV2MatrixSquareRoot:output:0MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddMatMul_3:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ ]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ o

Identity_1Identityinputs_1^NoOp*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:^ Z
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
L
Ф
C__inference_model_layer_call_and_return_conditional_losses_51639261
inputs_0
inputs_18
&dmgcn_0_matmul_readvariableop_resource:( 5
'dmgcn_0_biasadd_readvariableop_resource: 8
&dmgcn_1_matmul_readvariableop_resource:  5
'dmgcn_1_biasadd_readvariableop_resource: A
/dmdense_hidden_0_matmul_readvariableop_resource: >
0dmdense_hidden_0_biasadd_readvariableop_resource:A
/dmdense_hidden_1_matmul_readvariableop_resource:>
0dmdense_hidden_1_biasadd_readvariableop_resource:<
*dmdense_out_matmul_readvariableop_resource:9
+dmdense_out_biasadd_readvariableop_resource:
identityЂ'DMDense_Hidden_0/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_0/MatMul/ReadVariableOpЂ'DMDense_Hidden_1/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_1/MatMul/ReadVariableOpЂ"DMDense_OUT/BiasAdd/ReadVariableOpЂ!DMDense_OUT/MatMul/ReadVariableOpЂDMGCN_0/BiasAdd/ReadVariableOpЂDMGCN_0/MatMul/ReadVariableOpЂDMGCN_1/BiasAdd/ReadVariableOpЂDMGCN_1/MatMul/ReadVariableOph
DMGCN_0/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
DMGCN_0/SumSuminputs_1&DMGCN_0/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџP
DMGCN_0/diag/kConst*
_output_shapes
: *
dtype0*
value	B : `
DMGCN_0/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ`
DMGCN_0/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ_
DMGCN_0/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
DMGCN_0/diagMatrixDiagV3DMGCN_0/Sum:output:0DMGCN_0/diag/k:output:0DMGCN_0/diag/num_rows:output:0DMGCN_0/diag/num_cols:output:0#DMGCN_0/diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_0/MatrixInverseMatrixInverseDMGCN_0/diag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_0/MatrixSquareRootMatrixSquareRootDMGCN_0/MatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_0/MatMul/ReadVariableOpReadVariableOp&dmgcn_0_matmul_readvariableop_resource*
_output_shapes

:( *
dtype0
DMGCN_0/MatMulBatchMatMulV2inputs_0%DMGCN_0/MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/MatMul_1BatchMatMulV2!DMGCN_0/MatrixSquareRoot:output:0DMGCN_0/MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/MatMul_2BatchMatMulV2inputs_1DMGCN_0/MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/MatMul_3BatchMatMulV2!DMGCN_0/MatrixSquareRoot:output:0DMGCN_0/MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/BiasAdd/ReadVariableOpReadVariableOp'dmgcn_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
DMGCN_0/BiasAddBiasAddDMGCN_0/MatMul_3:output:0&DMGCN_0/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
DMGCN_0/ReluReluDMGCN_0/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ h
DMGCN_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
DMGCN_1/SumSuminputs_1&DMGCN_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџP
DMGCN_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : `
DMGCN_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ`
DMGCN_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ_
DMGCN_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
DMGCN_1/diagMatrixDiagV3DMGCN_1/Sum:output:0DMGCN_1/diag/k:output:0DMGCN_1/diag/num_rows:output:0DMGCN_1/diag/num_cols:output:0#DMGCN_1/diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_1/MatrixInverseMatrixInverseDMGCN_1/diag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_1/MatrixSquareRootMatrixSquareRootDMGCN_1/MatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_1/MatMul/ReadVariableOpReadVariableOp&dmgcn_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ё
DMGCN_1/MatMulBatchMatMulV2DMGCN_0/Relu:activations:0%DMGCN_1/MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/MatMul_1BatchMatMulV2!DMGCN_1/MatrixSquareRoot:output:0DMGCN_1/MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/MatMul_2BatchMatMulV2inputs_1DMGCN_1/MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/MatMul_3BatchMatMulV2!DMGCN_1/MatrixSquareRoot:output:0DMGCN_1/MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/BiasAdd/ReadVariableOpReadVariableOp'dmgcn_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
DMGCN_1/BiasAddBiasAddDMGCN_1/MatMul_3:output:0&DMGCN_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
DMGCN_1/ReluReluDMGCN_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ d
"DMGReduce_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
DMGReduce_1/MeanMeanDMGCN_1/Relu:activations:0+DMGReduce_1/Mean/reduction_indices:output:0*
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
:џџџџџџџџџr
DMDense_Hidden_0/ReluRelu!DMDense_Hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
&DMDense_Hidden_1/MatMul/ReadVariableOpReadVariableOp/dmdense_hidden_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ј
DMDense_Hidden_1/MatMulMatMul#DMDense_Hidden_0/Relu:activations:0.DMDense_Hidden_1/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџr
DMDense_Hidden_1/ReluRelu!DMDense_Hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!DMDense_OUT/MatMul/ReadVariableOpReadVariableOp*dmdense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
DMDense_OUT/MatMulMatMul#DMDense_Hidden_1/Relu:activations:0)DMDense_OUT/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџЗ
NoOpNoOp(^DMDense_Hidden_0/BiasAdd/ReadVariableOp'^DMDense_Hidden_0/MatMul/ReadVariableOp(^DMDense_Hidden_1/BiasAdd/ReadVariableOp'^DMDense_Hidden_1/MatMul/ReadVariableOp#^DMDense_OUT/BiasAdd/ReadVariableOp"^DMDense_OUT/MatMul/ReadVariableOp^DMGCN_0/BiasAdd/ReadVariableOp^DMGCN_0/MatMul/ReadVariableOp^DMGCN_1/BiasAdd/ReadVariableOp^DMGCN_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 2R
'DMDense_Hidden_0/BiasAdd/ReadVariableOp'DMDense_Hidden_0/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_0/MatMul/ReadVariableOp&DMDense_Hidden_0/MatMul/ReadVariableOp2R
'DMDense_Hidden_1/BiasAdd/ReadVariableOp'DMDense_Hidden_1/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_1/MatMul/ReadVariableOp&DMDense_Hidden_1/MatMul/ReadVariableOp2H
"DMDense_OUT/BiasAdd/ReadVariableOp"DMDense_OUT/BiasAdd/ReadVariableOp2F
!DMDense_OUT/MatMul/ReadVariableOp!DMDense_OUT/MatMul/ReadVariableOp2@
DMGCN_0/BiasAdd/ReadVariableOpDMGCN_0/BiasAdd/ReadVariableOp2>
DMGCN_0/MatMul/ReadVariableOpDMGCN_0/MatMul/ReadVariableOp2@
DMGCN_1/BiasAdd/ReadVariableOpDMGCN_1/BiasAdd/ReadVariableOp2>
DMGCN_1/MatMul/ReadVariableOpDMGCN_1/MatMul/ReadVariableOp:^ Z
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
­S
	
#__inference__wrapped_model_51638720
input_1
input_2>
,model_dmgcn_0_matmul_readvariableop_resource:( ;
-model_dmgcn_0_biasadd_readvariableop_resource: >
,model_dmgcn_1_matmul_readvariableop_resource:  ;
-model_dmgcn_1_biasadd_readvariableop_resource: G
5model_dmdense_hidden_0_matmul_readvariableop_resource: D
6model_dmdense_hidden_0_biasadd_readvariableop_resource:G
5model_dmdense_hidden_1_matmul_readvariableop_resource:D
6model_dmdense_hidden_1_biasadd_readvariableop_resource:B
0model_dmdense_out_matmul_readvariableop_resource:?
1model_dmdense_out_biasadd_readvariableop_resource:
identityЂ-model/DMDense_Hidden_0/BiasAdd/ReadVariableOpЂ,model/DMDense_Hidden_0/MatMul/ReadVariableOpЂ-model/DMDense_Hidden_1/BiasAdd/ReadVariableOpЂ,model/DMDense_Hidden_1/MatMul/ReadVariableOpЂ(model/DMDense_OUT/BiasAdd/ReadVariableOpЂ'model/DMDense_OUT/MatMul/ReadVariableOpЂ$model/DMGCN_0/BiasAdd/ReadVariableOpЂ#model/DMGCN_0/MatMul/ReadVariableOpЂ$model/DMGCN_1/BiasAdd/ReadVariableOpЂ#model/DMGCN_1/MatMul/ReadVariableOpn
#model/DMGCN_0/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
model/DMGCN_0/SumSuminput_2,model/DMGCN_0/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџV
model/DMGCN_0/diag/kConst*
_output_shapes
: *
dtype0*
value	B : f
model/DMGCN_0/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџf
model/DMGCN_0/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџe
 model/DMGCN_0/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/DMGCN_0/diagMatrixDiagV3model/DMGCN_0/Sum:output:0model/DMGCN_0/diag/k:output:0$model/DMGCN_0/diag/num_rows:output:0$model/DMGCN_0/diag/num_cols:output:0)model/DMGCN_0/diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGCN_0/MatrixInverseMatrixInversemodel/DMGCN_0/diag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
model/DMGCN_0/MatrixSquareRootMatrixSquareRoot$model/DMGCN_0/MatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
#model/DMGCN_0/MatMul/ReadVariableOpReadVariableOp,model_dmgcn_0_matmul_readvariableop_resource*
_output_shapes

:( *
dtype0
model/DMGCN_0/MatMulBatchMatMulV2input_1+model/DMGCN_0/MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ Ў
model/DMGCN_0/MatMul_1BatchMatMulV2'model/DMGCN_0/MatrixSquareRoot:output:0model/DMGCN_0/MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
model/DMGCN_0/MatMul_2BatchMatMulV2input_2model/DMGCN_0/MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ А
model/DMGCN_0/MatMul_3BatchMatMulV2'model/DMGCN_0/MatrixSquareRoot:output:0model/DMGCN_0/MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
$model/DMGCN_0/BiasAdd/ReadVariableOpReadVariableOp-model_dmgcn_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ў
model/DMGCN_0/BiasAddBiasAddmodel/DMGCN_0/MatMul_3:output:0,model/DMGCN_0/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ y
model/DMGCN_0/ReluRelumodel/DMGCN_0/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ n
#model/DMGCN_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
model/DMGCN_1/SumSuminput_2,model/DMGCN_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџV
model/DMGCN_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : f
model/DMGCN_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџf
model/DMGCN_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџe
 model/DMGCN_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
model/DMGCN_1/diagMatrixDiagV3model/DMGCN_1/Sum:output:0model/DMGCN_1/diag/k:output:0$model/DMGCN_1/diag/num_rows:output:0$model/DMGCN_1/diag/num_cols:output:0)model/DMGCN_1/diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
model/DMGCN_1/MatrixInverseMatrixInversemodel/DMGCN_1/diag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
model/DMGCN_1/MatrixSquareRootMatrixSquareRoot$model/DMGCN_1/MatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
#model/DMGCN_1/MatMul/ReadVariableOpReadVariableOp,model_dmgcn_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Г
model/DMGCN_1/MatMulBatchMatMulV2 model/DMGCN_0/Relu:activations:0+model/DMGCN_1/MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ Ў
model/DMGCN_1/MatMul_1BatchMatMulV2'model/DMGCN_1/MatrixSquareRoot:output:0model/DMGCN_1/MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
model/DMGCN_1/MatMul_2BatchMatMulV2input_2model/DMGCN_1/MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ А
model/DMGCN_1/MatMul_3BatchMatMulV2'model/DMGCN_1/MatrixSquareRoot:output:0model/DMGCN_1/MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
$model/DMGCN_1/BiasAdd/ReadVariableOpReadVariableOp-model_dmgcn_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ў
model/DMGCN_1/BiasAddBiasAddmodel/DMGCN_1/MatMul_3:output:0,model/DMGCN_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ y
model/DMGCN_1/ReluRelumodel/DMGCN_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ j
(model/DMGReduce_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ѕ
model/DMGReduce_1/MeanMean model/DMGCN_1/Relu:activations:01model/DMGReduce_1/Mean/reduction_indices:output:0*
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
:џџџџџџџџџ~
model/DMDense_Hidden_0/ReluRelu'model/DMDense_Hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
,model/DMDense_Hidden_1/MatMul/ReadVariableOpReadVariableOp5model_dmdense_hidden_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0К
model/DMDense_Hidden_1/MatMulMatMul)model/DMDense_Hidden_0/Relu:activations:04model/DMDense_Hidden_1/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџ~
model/DMDense_Hidden_1/ReluRelu'model/DMDense_Hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/DMDense_OUT/MatMul/ReadVariableOpReadVariableOp0model_dmdense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype0А
model/DMDense_OUT/MatMulMatMul)model/DMDense_Hidden_1/Relu:activations:0/model/DMDense_OUT/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџѓ
NoOpNoOp.^model/DMDense_Hidden_0/BiasAdd/ReadVariableOp-^model/DMDense_Hidden_0/MatMul/ReadVariableOp.^model/DMDense_Hidden_1/BiasAdd/ReadVariableOp-^model/DMDense_Hidden_1/MatMul/ReadVariableOp)^model/DMDense_OUT/BiasAdd/ReadVariableOp(^model/DMDense_OUT/MatMul/ReadVariableOp%^model/DMGCN_0/BiasAdd/ReadVariableOp$^model/DMGCN_0/MatMul/ReadVariableOp%^model/DMGCN_1/BiasAdd/ReadVariableOp$^model/DMGCN_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 2^
-model/DMDense_Hidden_0/BiasAdd/ReadVariableOp-model/DMDense_Hidden_0/BiasAdd/ReadVariableOp2\
,model/DMDense_Hidden_0/MatMul/ReadVariableOp,model/DMDense_Hidden_0/MatMul/ReadVariableOp2^
-model/DMDense_Hidden_1/BiasAdd/ReadVariableOp-model/DMDense_Hidden_1/BiasAdd/ReadVariableOp2\
,model/DMDense_Hidden_1/MatMul/ReadVariableOp,model/DMDense_Hidden_1/MatMul/ReadVariableOp2T
(model/DMDense_OUT/BiasAdd/ReadVariableOp(model/DMDense_OUT/BiasAdd/ReadVariableOp2R
'model/DMDense_OUT/MatMul/ReadVariableOp'model/DMDense_OUT/MatMul/ReadVariableOp2L
$model/DMGCN_0/BiasAdd/ReadVariableOp$model/DMGCN_0/BiasAdd/ReadVariableOp2J
#model/DMGCN_0/MatMul/ReadVariableOp#model/DMGCN_0/MatMul/ReadVariableOp2L
$model/DMGCN_1/BiasAdd/ReadVariableOp$model/DMGCN_1/BiasAdd/ReadVariableOp2J
#model/DMGCN_1/MatMul/ReadVariableOp#model/DMGCN_1/MatMul/ReadVariableOp:] Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
!
_user_specified_name	input_1:fb
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
!
_user_specified_name	input_2

u
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51639413
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
L
Ф
C__inference_model_layer_call_and_return_conditional_losses_51639326
inputs_0
inputs_18
&dmgcn_0_matmul_readvariableop_resource:( 5
'dmgcn_0_biasadd_readvariableop_resource: 8
&dmgcn_1_matmul_readvariableop_resource:  5
'dmgcn_1_biasadd_readvariableop_resource: A
/dmdense_hidden_0_matmul_readvariableop_resource: >
0dmdense_hidden_0_biasadd_readvariableop_resource:A
/dmdense_hidden_1_matmul_readvariableop_resource:>
0dmdense_hidden_1_biasadd_readvariableop_resource:<
*dmdense_out_matmul_readvariableop_resource:9
+dmdense_out_biasadd_readvariableop_resource:
identityЂ'DMDense_Hidden_0/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_0/MatMul/ReadVariableOpЂ'DMDense_Hidden_1/BiasAdd/ReadVariableOpЂ&DMDense_Hidden_1/MatMul/ReadVariableOpЂ"DMDense_OUT/BiasAdd/ReadVariableOpЂ!DMDense_OUT/MatMul/ReadVariableOpЂDMGCN_0/BiasAdd/ReadVariableOpЂDMGCN_0/MatMul/ReadVariableOpЂDMGCN_1/BiasAdd/ReadVariableOpЂDMGCN_1/MatMul/ReadVariableOph
DMGCN_0/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
DMGCN_0/SumSuminputs_1&DMGCN_0/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџP
DMGCN_0/diag/kConst*
_output_shapes
: *
dtype0*
value	B : `
DMGCN_0/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ`
DMGCN_0/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ_
DMGCN_0/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
DMGCN_0/diagMatrixDiagV3DMGCN_0/Sum:output:0DMGCN_0/diag/k:output:0DMGCN_0/diag/num_rows:output:0DMGCN_0/diag/num_cols:output:0#DMGCN_0/diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_0/MatrixInverseMatrixInverseDMGCN_0/diag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_0/MatrixSquareRootMatrixSquareRootDMGCN_0/MatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_0/MatMul/ReadVariableOpReadVariableOp&dmgcn_0_matmul_readvariableop_resource*
_output_shapes

:( *
dtype0
DMGCN_0/MatMulBatchMatMulV2inputs_0%DMGCN_0/MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/MatMul_1BatchMatMulV2!DMGCN_0/MatrixSquareRoot:output:0DMGCN_0/MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/MatMul_2BatchMatMulV2inputs_1DMGCN_0/MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/MatMul_3BatchMatMulV2!DMGCN_0/MatrixSquareRoot:output:0DMGCN_0/MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_0/BiasAdd/ReadVariableOpReadVariableOp'dmgcn_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
DMGCN_0/BiasAddBiasAddDMGCN_0/MatMul_3:output:0&DMGCN_0/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
DMGCN_0/ReluReluDMGCN_0/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ h
DMGCN_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
DMGCN_1/SumSuminputs_1&DMGCN_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџP
DMGCN_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : `
DMGCN_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ`
DMGCN_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ_
DMGCN_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ј
DMGCN_1/diagMatrixDiagV3DMGCN_1/Sum:output:0DMGCN_1/diag/k:output:0DMGCN_1/diag/num_rows:output:0DMGCN_1/diag/num_cols:output:0#DMGCN_1/diag/padding_value:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_1/MatrixInverseMatrixInverseDMGCN_1/diag:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_1/MatrixSquareRootMatrixSquareRootDMGCN_1/MatrixInverse:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
DMGCN_1/MatMul/ReadVariableOpReadVariableOp&dmgcn_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ё
DMGCN_1/MatMulBatchMatMulV2DMGCN_0/Relu:activations:0%DMGCN_1/MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/MatMul_1BatchMatMulV2!DMGCN_1/MatrixSquareRoot:output:0DMGCN_1/MatMul:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/MatMul_2BatchMatMulV2inputs_1DMGCN_1/MatMul_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/MatMul_3BatchMatMulV2!DMGCN_1/MatrixSquareRoot:output:0DMGCN_1/MatMul_2:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 
DMGCN_1/BiasAdd/ReadVariableOpReadVariableOp'dmgcn_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
DMGCN_1/BiasAddBiasAddDMGCN_1/MatMul_3:output:0&DMGCN_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ m
DMGCN_1/ReluReluDMGCN_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ d
"DMGReduce_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
DMGReduce_1/MeanMeanDMGCN_1/Relu:activations:0+DMGReduce_1/Mean/reduction_indices:output:0*
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
:џџџџџџџџџr
DMDense_Hidden_0/ReluRelu!DMDense_Hidden_0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
&DMDense_Hidden_1/MatMul/ReadVariableOpReadVariableOp/dmdense_hidden_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ј
DMDense_Hidden_1/MatMulMatMul#DMDense_Hidden_0/Relu:activations:0.DMDense_Hidden_1/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџr
DMDense_Hidden_1/ReluRelu!DMDense_Hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!DMDense_OUT/MatMul/ReadVariableOpReadVariableOp*dmdense_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
DMDense_OUT/MatMulMatMul#DMDense_Hidden_1/Relu:activations:0)DMDense_OUT/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџЗ
NoOpNoOp(^DMDense_Hidden_0/BiasAdd/ReadVariableOp'^DMDense_Hidden_0/MatMul/ReadVariableOp(^DMDense_Hidden_1/BiasAdd/ReadVariableOp'^DMDense_Hidden_1/MatMul/ReadVariableOp#^DMDense_OUT/BiasAdd/ReadVariableOp"^DMDense_OUT/MatMul/ReadVariableOp^DMGCN_0/BiasAdd/ReadVariableOp^DMGCN_0/MatMul/ReadVariableOp^DMGCN_1/BiasAdd/ReadVariableOp^DMGCN_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 2R
'DMDense_Hidden_0/BiasAdd/ReadVariableOp'DMDense_Hidden_0/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_0/MatMul/ReadVariableOp&DMDense_Hidden_0/MatMul/ReadVariableOp2R
'DMDense_Hidden_1/BiasAdd/ReadVariableOp'DMDense_Hidden_1/BiasAdd/ReadVariableOp2P
&DMDense_Hidden_1/MatMul/ReadVariableOp&DMDense_Hidden_1/MatMul/ReadVariableOp2H
"DMDense_OUT/BiasAdd/ReadVariableOp"DMDense_OUT/BiasAdd/ReadVariableOp2F
!DMDense_OUT/MatMul/ReadVariableOp!DMDense_OUT/MatMul/ReadVariableOp2@
DMGCN_0/BiasAdd/ReadVariableOpDMGCN_0/BiasAdd/ReadVariableOp2>
DMGCN_0/MatMul/ReadVariableOpDMGCN_0/MatMul/ReadVariableOp2@
DMGCN_1/BiasAdd/ReadVariableOpDMGCN_1/BiasAdd/ReadVariableOp2>
DMGCN_1/MatMul/ReadVariableOpDMGCN_1/MatMul/ReadVariableOp:^ Z
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
З


N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51639433
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
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
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
і
Z
.__inference_DMGReduce_1_layer_call_fn_51639406
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
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51638800`
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
М#
А
C__inference_model_layer_call_and_return_conditional_losses_51639001

inputs
inputs_1"
dmgcn_0_51638972:( 
dmgcn_0_51638974: "
dmgcn_1_51638978:  
dmgcn_1_51638980: +
dmdense_hidden_0_51638985: '
dmdense_hidden_0_51638987:+
dmdense_hidden_1_51638990:'
dmdense_hidden_1_51638992:&
dmdense_out_51638995:"
dmdense_out_51638997:
identityЂ(DMDense_Hidden_0/StatefulPartitionedCallЂ(DMDense_Hidden_1/StatefulPartitionedCallЂ#DMDense_OUT/StatefulPartitionedCallЂDMGCN_0/StatefulPartitionedCallЂDMGCN_1/StatefulPartitionedCallИ
DMGCN_0/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1dmgcn_0_51638972dmgcn_0_51638974*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51638754њ
DMGCN_1/StatefulPartitionedCallStatefulPartitionedCall(DMGCN_0/StatefulPartitionedCall:output:0(DMGCN_0/StatefulPartitionedCall:output:1dmgcn_1_51638978dmgcn_1_51638980*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџџџџџџџџџџ :'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8 *N
fIRG
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51638786
DMGReduce_1/PartitionedCallPartitionedCall(DMGCN_1/StatefulPartitionedCall:output:0(DMGCN_1/StatefulPartitionedCall:output:1*
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
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51638800И
(DMDense_Hidden_0/StatefulPartitionedCallStatefulPartitionedCall$DMGReduce_1/PartitionedCall:output:0dmdense_hidden_0_51638985dmdense_hidden_0_51638987*
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
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51638813Х
(DMDense_Hidden_1/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_0/StatefulPartitionedCall:output:0dmdense_hidden_1_51638990dmdense_hidden_1_51638992*
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
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51638830Б
#DMDense_OUT/StatefulPartitionedCallStatefulPartitionedCall1DMDense_Hidden_1/StatefulPartitionedCall:output:0dmdense_out_51638995dmdense_out_51638997*
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
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51638846{
IdentityIdentity,DMDense_OUT/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp)^DMDense_Hidden_0/StatefulPartitionedCall)^DMDense_Hidden_1/StatefulPartitionedCall$^DMDense_OUT/StatefulPartitionedCall ^DMGCN_0/StatefulPartitionedCall ^DMGCN_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџџџџџџџџџџ(:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : : : : : : : 2T
(DMDense_Hidden_0/StatefulPartitionedCall(DMDense_Hidden_0/StatefulPartitionedCall2T
(DMDense_Hidden_1/StatefulPartitionedCall(DMDense_Hidden_1/StatefulPartitionedCall2J
#DMDense_OUT/StatefulPartitionedCall#DMDense_OUT/StatefulPartitionedCall2B
DMGCN_0/StatefulPartitionedCallDMGCN_0/StatefulPartitionedCall2B
DMGCN_1/StatefulPartitionedCallDMGCN_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ(
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ПL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultњ
H
input_1=
serving_default_input_1:0џџџџџџџџџџџџџџџџџџ(
Q
input_2F
serving_default_input_2:0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ?
DMDense_OUT0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:л­
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
м
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
DMGCN_0_weight
w
DMGCN_0_bias
bias"
_tf_keras_layer
м
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
DMGCN_1_weight
w
 DMGCN_1_bias
 bias"
_tf_keras_layer
Ѕ
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
ѓ
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-DMDense_Hidden_0_weight

-weight
.DMDense_Hidden_0_bias
.bias"
_tf_keras_layer
ѓ
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5DMDense_Hidden_1_weight

5weight
6DMDense_Hidden_1_bias
6bias"
_tf_keras_layer
щ
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=DMDense_OUT_weight

=weight
>DMDense_OUT_bias
>bias"
_tf_keras_layer
f
0
1
2
 3
-4
.5
56
67
=8
>9"
trackable_list_wrapper
f
0
1
2
 3
-4
.5
56
67
=8
>9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
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
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_32ы
(__inference_model_layer_call_fn_51638876
(__inference_model_layer_call_fn_51639170
(__inference_model_layer_call_fn_51639196
(__inference_model_layer_call_fn_51639050Р
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
 zDtrace_0zEtrace_1zFtrace_2zGtrace_3
Т
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32з
C__inference_model_layer_call_and_return_conditional_losses_51639261
C__inference_model_layer_call_and_return_conditional_losses_51639326
C__inference_model_layer_call_and_return_conditional_losses_51639083
C__inference_model_layer_call_and_return_conditional_losses_51639116Р
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
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
зBд
#__inference__wrapped_model_51638720input_1input_2"
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
Lserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
Rtrace_02б
*__inference_DMGCN_0_layer_call_fn_51639338Ђ
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
 zRtrace_0

Strace_02ь
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51639363Ђ
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
 zStrace_0
(:&( 2DMGCN_0/DMGCN_0_weight
":  2DMGCN_0/DMGCN_0_bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
Ytrace_02б
*__inference_DMGCN_1_layer_call_fn_51639375Ђ
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
 zYtrace_0

Ztrace_02ь
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51639400Ђ
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
 zZtrace_0
(:&  2DMGCN_1/DMGCN_1_weight
":  2DMGCN_1/DMGCN_1_bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ђ
`trace_02е
.__inference_DMGReduce_1_layer_call_fn_51639406Ђ
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
 z`trace_0

atrace_02№
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51639413Ђ
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
 zatrace_0
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
§
gtrace_02р
3__inference_DMDense_Hidden_0_layer_call_fn_51639422Ј
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
 zgtrace_0

htrace_02ћ
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51639433Ј
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
 zhtrace_0
::8 2(DMDense_Hidden_0/DMDense_Hidden_0_weight
4:22&DMDense_Hidden_0/DMDense_Hidden_0_bias
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
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
§
ntrace_02р
3__inference_DMDense_Hidden_1_layer_call_fn_51639442Ј
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
 zntrace_0

otrace_02ћ
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51639453Ј
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
 zotrace_0
::82(DMDense_Hidden_1/DMDense_Hidden_1_weight
4:22&DMDense_Hidden_1/DMDense_Hidden_1_bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
ј
utrace_02л
.__inference_DMDense_OUT_layer_call_fn_51639462Ј
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
 zutrace_0

vtrace_02і
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51639472Ј
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
 zvtrace_0
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
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
(__inference_model_layer_call_fn_51638876input_1input_2"Р
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
(__inference_model_layer_call_fn_51639170inputs/0inputs/1"Р
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
(__inference_model_layer_call_fn_51639196inputs/0inputs/1"Р
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
B
(__inference_model_layer_call_fn_51639050input_1input_2"Р
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
C__inference_model_layer_call_and_return_conditional_losses_51639261inputs/0inputs/1"Р
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
C__inference_model_layer_call_and_return_conditional_losses_51639326inputs/0inputs/1"Р
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
B
C__inference_model_layer_call_and_return_conditional_losses_51639083input_1input_2"Р
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
B
C__inference_model_layer_call_and_return_conditional_losses_51639116input_1input_2"Р
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
дBб
&__inference_signature_wrapper_51639144input_1input_2"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
*__inference_DMGCN_0_layer_call_fn_51639338inputs/0inputs/1"Ђ
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
B
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51639363inputs/0inputs/1"Ђ
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
ъBч
*__inference_DMGCN_1_layer_call_fn_51639375inputs/0inputs/1"Ђ
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
B
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51639400inputs/0inputs/1"Ђ
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
юBы
.__inference_DMGReduce_1_layer_call_fn_51639406inputs/0inputs/1"Ђ
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
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51639413inputs/0inputs/1"Ђ
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
3__inference_DMDense_Hidden_0_layer_call_fn_51639422input_tensor"Ј
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
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51639433input_tensor"Ј
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
3__inference_DMDense_Hidden_1_layer_call_fn_51639442input_tensor"Ј
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
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51639453input_tensor"Ј
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
.__inference_DMDense_OUT_layer_call_fn_51639462input_tensor"Ј
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
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51639472input_tensor"Ј
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
N
y	variables
z	keras_api
	{total
	|count"
_tf_keras_metric
`
}	variables
~	keras_api
	total

count

_fn_kwargs"
_tf_keras_metric
.
{0
|1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
/
0
1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperД
N__inference_DMDense_Hidden_0_layer_call_and_return_conditional_losses_51639433b-.5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 
3__inference_DMDense_Hidden_0_layer_call_fn_51639422U-.5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ 
Њ "џџџџџџџџџД
N__inference_DMDense_Hidden_1_layer_call_and_return_conditional_losses_51639453b565Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
3__inference_DMDense_Hidden_1_layer_call_fn_51639442U565Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "џџџџџџџџџЏ
I__inference_DMDense_OUT_layer_call_and_return_conditional_losses_51639472b=>5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_DMDense_OUT_layer_call_fn_51639462U=>5Ђ2
+Ђ(
&#
input_tensorџџџџџџџџџ
Њ "џџџџџџџџџН
E__inference_DMGCN_0_layer_call_and_return_conditional_losses_51639363ѓ}Ђz
sЂp
nk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "nЂk
dЂa
*'
0/0џџџџџџџџџџџџџџџџџџ 
30
0/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
*__inference_DMGCN_0_layer_call_fn_51639338х}Ђz
sЂp
nk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "`Ђ]
(%
0џџџџџџџџџџџџџџџџџџ 
1.
1'џџџџџџџџџџџџџџџџџџџџџџџџџџџН
E__inference_DMGCN_1_layer_call_and_return_conditional_losses_51639400ѓ }Ђz
sЂp
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "nЂk
dЂa
*'
0/0џџџџџџџџџџџџџџџџџџ 
30
0/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
*__inference_DMGCN_1_layer_call_fn_51639375х }Ђz
sЂp
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "`Ђ]
(%
0џџџџџџџџџџџџџџџџџџ 
1.
1'џџџџџџџџџџџџџџџџџџџџџџџџџџџє
I__inference_DMGReduce_1_layer_call_and_return_conditional_losses_51639413І}Ђz
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
.__inference_DMGReduce_1_layer_call_fn_51639406}Ђz
sЂp
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ 
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "џџџџџџџџџ ь
#__inference__wrapped_model_51638720Ф
 -.56=>{Ђx
qЂn
lЂi
.+
input_1џџџџџџџџџџџџџџџџџџ(
74
input_2'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "9Њ6
4
DMDense_OUT%"
DMDense_OUTџџџџџџџџџ
C__inference_model_layer_call_and_return_conditional_losses_51639083К
 -.56=>Ђ
yЂv
lЂi
.+
input_1џџџџџџџџџџџџџџџџџџ(
74
input_2'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
C__inference_model_layer_call_and_return_conditional_losses_51639116К
 -.56=>Ђ
yЂv
lЂi
.+
input_1џџџџџџџџџџџџџџџџџџ(
74
input_2'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
C__inference_model_layer_call_and_return_conditional_losses_51639261М
 -.56=>Ђ
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
 
C__inference_model_layer_call_and_return_conditional_losses_51639326М
 -.56=>Ђ
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
 к
(__inference_model_layer_call_fn_51638876­
 -.56=>Ђ
yЂv
lЂi
.+
input_1џџџџџџџџџџџџџџџџџџ(
74
input_2'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџк
(__inference_model_layer_call_fn_51639050­
 -.56=>Ђ
yЂv
lЂi
.+
input_1џџџџџџџџџџџџџџџџџџ(
74
input_2'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџм
(__inference_model_layer_call_fn_51639170Џ
 -.56=>Ђ
{Ђx
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџм
(__inference_model_layer_call_fn_51639196Џ
 -.56=>Ђ
{Ђx
nЂk
/,
inputs/0џџџџџџџџџџџџџџџџџџ(
85
inputs/1'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
&__inference_signature_wrapper_51639144и
 -.56=>Ђ
Ђ 
Њ
9
input_1.+
input_1џџџџџџџџџџџџџџџџџџ(
B
input_274
input_2'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"9Њ6
4
DMDense_OUT%"
DMDense_OUTџџџџџџџџџ