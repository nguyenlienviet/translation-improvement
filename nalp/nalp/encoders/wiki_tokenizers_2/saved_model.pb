??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
?
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0?????????"
value_indexint(0?????????"+

vocab_sizeint?????????(0?????????"
	delimiterstring	"
offsetint ?
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
H
ShardedFilename
basename	
shard

num_shards
filename
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
R
Variable/AssignAssignVariableOpVariableasset_path_initializer*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
n

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_name
Variable_1
g
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:??*
dtype0
?

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*O
shared_name@>hash_table_/content/drive/MyDrive/CSC 582/wiki_vocab.txt_-2_-1*
value_dtype0	
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
?

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
X
Variable_2/AssignAssignVariableOp
Variable_2asset_path_initializer_1*
dtype0
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
P
ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R
e
ReadVariableOpReadVariableOp
Variable_2^Variable_2/Assign*
_output_shapes
: *
dtype0
?
StatefulPartitionedCallStatefulPartitionedCallReadVariableOp
hash_table*
Tin
2*
Tout
2*
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
GPU 2J 8? *"
fR
__inference_<lambda>_5358
L
NoOpNoOp^StatefulPartitionedCall^Variable/Assign^Variable_2/Assign
?
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?

en

signatures
A
	tokenizer
_reserved_tokens
_vocab_path
	vocab
 
0
_basic_tokenizer
_wordpiece_tokenizer
 
 
CA
VARIABLE_VALUE
Variable_1#en/vocab/.ATTRIBUTES/VARIABLE_VALUE
 

	_vocab_lookup_table


_initializer

	_filename
 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_1/Read/ReadVariableOpConst_3*
Tin
2*
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
GPU 2J 8? *&
f!R
__inference__traced_save_5396
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_1*
Tin
2*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_5409??
?
?
rRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4911?
?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
?
mRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2o
mRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp?
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2u
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?
?
9RaggedConcat_assert_equal_3_Assert_AssertGuard_false_5266Y
Uraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_all
i
eraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_1_strided_slice_4	g
craggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4	=
9raggedconcat_assert_equal_3_assert_assertguard_identity_1
??5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert?
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'Input tensors have incompatible shapes.2>
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_0?
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2>
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_1?
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*I
value@B> B8x (RaggedConcat/RaggedFromTensor_1/strided_slice_4:0) = 2>
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_2?
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = 2>
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_4?
5RaggedConcat/assert_equal_3/Assert/AssertGuard/AssertAssertUraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_allERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_0:output:0ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_1:output:0ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_2:output:0eraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_1_strided_slice_4ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_4:output:0craggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4*
T

2		*
_output_shapes
 27
5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert?
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentityUraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_all6^RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 29
7RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity?
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity:output:06^RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2;
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1"
9raggedconcat_assert_equal_3_assert_assertguard_identity_1BRaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2n
5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
<
__inference_get_vocab_path_4588
unknown
identityJ
IdentityIdentityunknown*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
BRaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_false_4949k
graggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_all
h
draggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice_1	f
braggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice	F
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1
??>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert?
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor2G
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_0?
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2G
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_1?
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_2/strided_slice_1:0) = 2G
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_2?
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_2/strided_slice:0) = 2G
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_4?
>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_allNRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice_1NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice*
T

2		*
_output_shapes
 2@
>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert?
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_all?^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity:output:0?^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2D
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1"?
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
BRaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_false_5076k
graggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_all
h
draggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice_1	f
braggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice	F
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1
??>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert?
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor2G
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_0?
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2G
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_1?
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_3/strided_slice_1:0) = 2G
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_2?
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_3/strided_slice:0) = 2G
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_4?
>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_allNRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice_1NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice*
T

2		*
_output_shapes
 2@
>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert?
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_all?^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity:output:0?^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2D
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1"?
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
4
$__inference_get_reserved_tokens_4583
identitys
ConstConst*
_output_shapes
:*
dtype0*1
value(B&B[PAD]B[UNK]B[START]B[END]2
ConstU
IdentityIdentityConst:output:0*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
__inference_tokenize_5334
stringsm
iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tableu
qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_value	

fill_value	
fill_1_value	
identity	

identity_1	??.RaggedConcat/assert_equal_1/Assert/AssertGuard?.RaggedConcat/assert_equal_3/Assert/AssertGuard?ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard?QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard?QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard?QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard?cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2?aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2?VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetst
CaseFoldUTF8/CaseFoldUTF8CaseFoldUTF8strings*#
_output_shapes
:?????????2
CaseFoldUTF8/CaseFoldUTF8?
NormalizeUTF8/NormalizeUTF8NormalizeUTF8"CaseFoldUTF8/CaseFoldUTF8:output:0*#
_output_shapes
:?????????*
normalization_formNFD2
NormalizeUTF8/NormalizeUTF8?
StaticRegexReplaceStaticRegexReplace$NormalizeUTF8/NormalizeUTF8:output:0*#
_output_shapes
:?????????*
pattern\p{Mn}*
rewrite 2
StaticRegexReplace?
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:?????????*
pattern\p{Cc}|\p{Cf}*
rewrite 2
StaticRegexReplace_1[
ShapeShapeStaticRegexReplace_1:output:0*
T0*
_output_shapes
:2
ShapeX
CastCastShape:output:0*

DstT0	*

SrcT0*
_output_shapes
:2
Castq
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shape?
ReshapeReshapeStaticRegexReplace_1:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
Reshape?
)RegexSplitWithOffsets/delim_regex_patternConst*
_output_shapes
: *
dtype0*?
value?B? B?(\s+|[!-/]|[:-@]|[\[-`]|[{-~]|[\p{P}]|[\x{4E00}-\x{9FFF}]|[\x{3400}-\x{4DBF}]|[\x{20000}-\x{2A6DF}]|[\x{2A700}-\x{2B73F}]|[\x{2B740}-\x{2B81F}]|[\x{2B820}-\x{2CEAF}]|[\x{F900}-\x{FAFF}]|[\x{2F800}-\x{2FA1F}])2+
)RegexSplitWithOffsets/delim_regex_pattern?
.RegexSplitWithOffsets/keep_delim_regex_patternConst*
_output_shapes
: *
dtype0*?
value?B? B?([!-/]|[:-@]|[\[-`]|[{-~]|[\p{P}]|[\x{4E00}-\x{9FFF}]|[\x{3400}-\x{4DBF}]|[\x{20000}-\x{2A6DF}]|[\x{2A700}-\x{2B73F}]|[\x{2B740}-\x{2B81F}]|[\x{2B820}-\x{2CEAF}]|[\x{F900}-\x{FAFF}]|[\x{2F800}-\x{2FA1F}])20
.RegexSplitWithOffsets/keep_delim_regex_pattern?
RegexSplitWithOffsetsRegexSplitWithOffsetsReshape:output:02RegexSplitWithOffsets/delim_regex_pattern:output:07RegexSplitWithOffsets/keep_delim_regex_pattern:output:0*P
_output_shapes>
<:?????????:?????????:?????????:?????????2
RegexSplitWithOffsets?
>RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :2@
>RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/rank?
?RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
:2A
?RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/Shape?
hRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2j
hRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_type?
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*
_output_shapes
 2[
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok?
ARaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
ARaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack?
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1?
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2?
;RaggedFromRowSplits/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0JRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack:output:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1:output:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2=
;RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice?
3RaggedFromRowSplits/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 25
3RaggedFromRowSplits/RowPartitionFromRowSplits/Const?
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/EqualEqualDRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0<RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: 2D
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Equal?
ARaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2C
ARaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Rank?
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2J
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/start?
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2J
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/delta?
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/rangeRangeQRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0JRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0QRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: 2D
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range?
@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/AllAllFRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: 2B
@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All?
IRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2K
IRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const?
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2M
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1?
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = 2M
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2?
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = 2M
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3?
ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfIRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All:output:0IRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All:output:0DRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0<RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *m
else_branch^R\
ZRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4650*
output_shapes
: *l
then_branch]R[
YRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_46492Q
ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityXRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2Z
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack?
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1?
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2?
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2?
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1?
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack?
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2G
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1?
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2?
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2?
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2?
1RaggedFromRowSplits/RowPartitionFromRowSplits/subSubFRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1:output:0FRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:?????????23
1RaggedFromRowSplits/RowPartitionFromRowSplits/sub?
GRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2I
GRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/Const?
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualPRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/Const:output:05RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:?????????2_
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual?
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2[
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const?
WRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: 2Y
WRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All?
`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2b
`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const?
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2d
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1?
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = 2d
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2?
fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIf`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:05RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0P^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branchuRs
qRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4686*
output_shapes
: *?
then_branchtRr
pRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_46852h
fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityoRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2q
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
@RaggedFromRowSplits/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0Y^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityp^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityZ^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:?????????2B
@RaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency?
RaggedFromRowSplits/ShapeShapeRegexSplitWithOffsets:tokens:0*
T0*
_output_shapes
:*
out_type0	2
RaggedFromRowSplits/Shape?
'RaggedFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'RaggedFromRowSplits/strided_slice/stack?
)RaggedFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedFromRowSplits/strided_slice/stack_1?
)RaggedFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedFromRowSplits/strided_slice/stack_2?
!RaggedFromRowSplits/strided_sliceStridedSlice"RaggedFromRowSplits/Shape:output:00RaggedFromRowSplits/strided_slice/stack:output:02RaggedFromRowSplits/strided_slice/stack_1:output:02RaggedFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2#
!RaggedFromRowSplits/strided_slice?
)RaggedFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)RaggedFromRowSplits/strided_slice_1/stack?
+RaggedFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+RaggedFromRowSplits/strided_slice_1/stack_1?
+RaggedFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+RaggedFromRowSplits/strided_slice_1/stack_2?
#RaggedFromRowSplits/strided_slice_1StridedSliceIRaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:02RaggedFromRowSplits/strided_slice_1/stack:output:04RaggedFromRowSplits/strided_slice_1/stack_1:output:04RaggedFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2%
#RaggedFromRowSplits/strided_slice_1?
(RaggedFromRowSplits/assert_equal_1/EqualEqual,RaggedFromRowSplits/strided_slice_1:output:0*RaggedFromRowSplits/strided_slice:output:0*
T0	*
_output_shapes
: 2*
(RaggedFromRowSplits/assert_equal_1/Equal?
'RaggedFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2)
'RaggedFromRowSplits/assert_equal_1/Rank?
.RaggedFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 20
.RaggedFromRowSplits/assert_equal_1/range/start?
.RaggedFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :20
.RaggedFromRowSplits/assert_equal_1/range/delta?
(RaggedFromRowSplits/assert_equal_1/rangeRange7RaggedFromRowSplits/assert_equal_1/range/start:output:00RaggedFromRowSplits/assert_equal_1/Rank:output:07RaggedFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: 2*
(RaggedFromRowSplits/assert_equal_1/range?
&RaggedFromRowSplits/assert_equal_1/AllAll,RaggedFromRowSplits/assert_equal_1/Equal:z:01RaggedFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: 2(
&RaggedFromRowSplits/assert_equal_1/All?
/RaggedFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor21
/RaggedFromRowSplits/assert_equal_1/Assert/Const?
1RaggedFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:23
1RaggedFromRowSplits/assert_equal_1/Assert/Const_1?
1RaggedFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (RaggedFromRowSplits/strided_slice_1:0) = 23
1RaggedFromRowSplits/assert_equal_1/Assert/Const_2?
1RaggedFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (RaggedFromRowSplits/strided_slice:0) = 23
1RaggedFromRowSplits/assert_equal_1/Assert/Const_3?
5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardIf/RaggedFromRowSplits/assert_equal_1/All:output:0/RaggedFromRowSplits/assert_equal_1/All:output:0,RaggedFromRowSplits/strided_slice_1:output:0*RaggedFromRowSplits/strided_slice:output:0g^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *S
else_branchDRB
@RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4723*
output_shapes
: *R
then_branchCRA
?RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_472227
5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard?
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2@
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
-RaggedFromRowSplits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :2/
-RaggedFromRowSplits/assert_rank_at_least/rank?
.RaggedFromRowSplits/assert_rank_at_least/ShapeShapeRegexSplitWithOffsets:tokens:0*
T0*
_output_shapes
:20
.RaggedFromRowSplits/assert_rank_at_least/Shape?
WRaggedFromRowSplits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2Y
WRaggedFromRowSplits/assert_rank_at_least/assert_type/statically_determined_correct_type?
HRaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 2J
HRaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_ok?
&RaggedFromRowSplits/control_dependencyIdentityIRaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:0?^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityI^RaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:?????????2(
&RaggedFromRowSplits/control_dependency?
@RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :2B
@RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/rank?
ARaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
:2C
ARaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/Shape?
jRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2l
jRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_type?
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*
_output_shapes
 2]
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok?
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack?
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1?
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2?
=RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2?
=RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice?
5RaggedFromRowSplits_1/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 27
5RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const?
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: 2F
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Equal?
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2E
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Rank?
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2L
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/start?
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2L
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/delta?
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: 2F
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range?
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: 2D
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All?
KRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2M
KRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const?
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2O
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1?
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = 2O
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2?
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = 2O
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3?
QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:06^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4763*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_47622S
QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack?
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1?
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2?
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2A
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1?
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack?
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2I
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1?
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2?
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2A
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2?
3RaggedFromRowSplits_1/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:?????????25
3RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub?
IRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2K
IRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/Const?
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:?????????2a
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual?
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2]
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const?
YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: 2[
YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All?
bRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2d
bRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const?
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2f
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1?
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = 2f
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2?
hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branchwRu
sRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4799*
output_shapes
: *?
then_branchvRt
rRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_47982j
hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0[^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:?????????2D
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency?
RaggedFromRowSplits_1/ShapeShape%RegexSplitWithOffsets:begin_offsets:0*
T0	*
_output_shapes
:*
out_type0	2
RaggedFromRowSplits_1/Shape?
)RaggedFromRowSplits_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)RaggedFromRowSplits_1/strided_slice/stack?
+RaggedFromRowSplits_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+RaggedFromRowSplits_1/strided_slice/stack_1?
+RaggedFromRowSplits_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+RaggedFromRowSplits_1/strided_slice/stack_2?
#RaggedFromRowSplits_1/strided_sliceStridedSlice$RaggedFromRowSplits_1/Shape:output:02RaggedFromRowSplits_1/strided_slice/stack:output:04RaggedFromRowSplits_1/strided_slice/stack_1:output:04RaggedFromRowSplits_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2%
#RaggedFromRowSplits_1/strided_slice?
+RaggedFromRowSplits_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+RaggedFromRowSplits_1/strided_slice_1/stack?
-RaggedFromRowSplits_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-RaggedFromRowSplits_1/strided_slice_1/stack_1?
-RaggedFromRowSplits_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-RaggedFromRowSplits_1/strided_slice_1/stack_2?
%RaggedFromRowSplits_1/strided_slice_1StridedSliceKRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_1/strided_slice_1/stack:output:06RaggedFromRowSplits_1/strided_slice_1/stack_1:output:06RaggedFromRowSplits_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%RaggedFromRowSplits_1/strided_slice_1?
*RaggedFromRowSplits_1/assert_equal_1/EqualEqual.RaggedFromRowSplits_1/strided_slice_1:output:0,RaggedFromRowSplits_1/strided_slice:output:0*
T0	*
_output_shapes
: 2,
*RaggedFromRowSplits_1/assert_equal_1/Equal?
)RaggedFromRowSplits_1/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2+
)RaggedFromRowSplits_1/assert_equal_1/Rank?
0RaggedFromRowSplits_1/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0RaggedFromRowSplits_1/assert_equal_1/range/start?
0RaggedFromRowSplits_1/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0RaggedFromRowSplits_1/assert_equal_1/range/delta?
*RaggedFromRowSplits_1/assert_equal_1/rangeRange9RaggedFromRowSplits_1/assert_equal_1/range/start:output:02RaggedFromRowSplits_1/assert_equal_1/Rank:output:09RaggedFromRowSplits_1/assert_equal_1/range/delta:output:0*
_output_shapes
: 2,
*RaggedFromRowSplits_1/assert_equal_1/range?
(RaggedFromRowSplits_1/assert_equal_1/AllAll.RaggedFromRowSplits_1/assert_equal_1/Equal:z:03RaggedFromRowSplits_1/assert_equal_1/range:output:0*
_output_shapes
: 2*
(RaggedFromRowSplits_1/assert_equal_1/All?
1RaggedFromRowSplits_1/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor23
1RaggedFromRowSplits_1/assert_equal_1/Assert/Const?
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:25
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_1?
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_1/strided_slice_1:0) = 25
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_2?
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_1/strided_slice:0) = 25
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_3?
7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_1/assert_equal_1/All:output:01RaggedFromRowSplits_1/assert_equal_1/All:output:0.RaggedFromRowSplits_1/strided_slice_1:output:0,RaggedFromRowSplits_1/strided_slice:output:0i^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_4836*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_483529
7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard?
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity?
/RaggedFromRowSplits_1/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :21
/RaggedFromRowSplits_1/assert_rank_at_least/rank?
0RaggedFromRowSplits_1/assert_rank_at_least/ShapeShape%RegexSplitWithOffsets:begin_offsets:0*
T0	*
_output_shapes
:22
0RaggedFromRowSplits_1/assert_rank_at_least/Shape?
YRaggedFromRowSplits_1/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2[
YRaggedFromRowSplits_1/assert_rank_at_least/assert_type/statically_determined_correct_type?
JRaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 2L
JRaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_ok?
(RaggedFromRowSplits_1/control_dependencyIdentityKRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:?????????2*
(RaggedFromRowSplits_1/control_dependency?
@RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :2B
@RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/rank?
ARaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
:2C
ARaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/Shape?
jRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2l
jRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_type?
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*
_output_shapes
 2]
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok?
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack?
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1?
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2?
=RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2?
=RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice?
5RaggedFromRowSplits_2/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 27
5RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const?
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: 2F
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Equal?
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2E
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Rank?
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2L
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/start?
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2L
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/delta?
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: 2F
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range?
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: 2D
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All?
KRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2M
KRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const?
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2O
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1?
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = 2O
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2?
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = 2O
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3?
QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:08^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4876*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_48752S
QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack?
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1?
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2?
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2A
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1?
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack?
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2I
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1?
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2?
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2A
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2?
3RaggedFromRowSplits_2/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:?????????25
3RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub?
IRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2K
IRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/Const?
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:?????????2a
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual?
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2]
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const?
YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: 2[
YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All?
bRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2d
bRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const?
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2f
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1?
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = 2f
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2?
hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branchwRu
sRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4912*
output_shapes
: *?
then_branchvRt
rRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_49112j
hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0[^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:?????????2D
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency?
RaggedFromRowSplits_2/ShapeShape#RegexSplitWithOffsets:end_offsets:0*
T0	*
_output_shapes
:*
out_type0	2
RaggedFromRowSplits_2/Shape?
)RaggedFromRowSplits_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)RaggedFromRowSplits_2/strided_slice/stack?
+RaggedFromRowSplits_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+RaggedFromRowSplits_2/strided_slice/stack_1?
+RaggedFromRowSplits_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+RaggedFromRowSplits_2/strided_slice/stack_2?
#RaggedFromRowSplits_2/strided_sliceStridedSlice$RaggedFromRowSplits_2/Shape:output:02RaggedFromRowSplits_2/strided_slice/stack:output:04RaggedFromRowSplits_2/strided_slice/stack_1:output:04RaggedFromRowSplits_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2%
#RaggedFromRowSplits_2/strided_slice?
+RaggedFromRowSplits_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+RaggedFromRowSplits_2/strided_slice_1/stack?
-RaggedFromRowSplits_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-RaggedFromRowSplits_2/strided_slice_1/stack_1?
-RaggedFromRowSplits_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-RaggedFromRowSplits_2/strided_slice_1/stack_2?
%RaggedFromRowSplits_2/strided_slice_1StridedSliceKRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_2/strided_slice_1/stack:output:06RaggedFromRowSplits_2/strided_slice_1/stack_1:output:06RaggedFromRowSplits_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%RaggedFromRowSplits_2/strided_slice_1?
*RaggedFromRowSplits_2/assert_equal_1/EqualEqual.RaggedFromRowSplits_2/strided_slice_1:output:0,RaggedFromRowSplits_2/strided_slice:output:0*
T0	*
_output_shapes
: 2,
*RaggedFromRowSplits_2/assert_equal_1/Equal?
)RaggedFromRowSplits_2/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2+
)RaggedFromRowSplits_2/assert_equal_1/Rank?
0RaggedFromRowSplits_2/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0RaggedFromRowSplits_2/assert_equal_1/range/start?
0RaggedFromRowSplits_2/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0RaggedFromRowSplits_2/assert_equal_1/range/delta?
*RaggedFromRowSplits_2/assert_equal_1/rangeRange9RaggedFromRowSplits_2/assert_equal_1/range/start:output:02RaggedFromRowSplits_2/assert_equal_1/Rank:output:09RaggedFromRowSplits_2/assert_equal_1/range/delta:output:0*
_output_shapes
: 2,
*RaggedFromRowSplits_2/assert_equal_1/range?
(RaggedFromRowSplits_2/assert_equal_1/AllAll.RaggedFromRowSplits_2/assert_equal_1/Equal:z:03RaggedFromRowSplits_2/assert_equal_1/range:output:0*
_output_shapes
: 2*
(RaggedFromRowSplits_2/assert_equal_1/All?
1RaggedFromRowSplits_2/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor23
1RaggedFromRowSplits_2/assert_equal_1/Assert/Const?
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:25
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_1?
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_2/strided_slice_1:0) = 25
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_2?
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_2/strided_slice:0) = 25
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_3?
7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_2/assert_equal_1/All:output:01RaggedFromRowSplits_2/assert_equal_1/All:output:0.RaggedFromRowSplits_2/strided_slice_1:output:0,RaggedFromRowSplits_2/strided_slice:output:0i^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_false_4949*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_true_494829
7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard?
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity?
/RaggedFromRowSplits_2/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :21
/RaggedFromRowSplits_2/assert_rank_at_least/rank?
0RaggedFromRowSplits_2/assert_rank_at_least/ShapeShape#RegexSplitWithOffsets:end_offsets:0*
T0	*
_output_shapes
:22
0RaggedFromRowSplits_2/assert_rank_at_least/Shape?
YRaggedFromRowSplits_2/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2[
YRaggedFromRowSplits_2/assert_rank_at_least/assert_type/statically_determined_correct_type?
JRaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 2L
JRaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_ok?
(RaggedFromRowSplits_2/control_dependencyIdentityKRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:?????????2*
(RaggedFromRowSplits_2/control_dependency?
VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsWordpieceTokenizeWithOffsetsRegexSplitWithOffsets:tokens:0iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_table*P
_output_shapes>
<:?????????:?????????:?????????:?????????*
max_bytes_per_wordd*)
output_row_partition_type
row_splits*
suffix_indicator##*
unknown_token[UNK]*
use_unknown_token(2X
VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets?
QWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/hash_bucketStringToHashBucketFastfWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_values:0*#
_output_shapes
:?????????*
num_buckets2S
QWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/hash_bucket?
cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tablefWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_values:0qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_valueW^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets*	
Tin0*

Tout0	*#
_output_shapes
:?????????2e
cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2?
aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tabled^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2*
_output_shapes
: 2c
aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2?
IWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/AddAddZWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/hash_bucket:output:0hWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*#
_output_shapes
:?????????2K
IWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/Add?
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/NotEqualNotEquallWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2:values:0qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*#
_output_shapes
:?????????2P
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/NotEqual?
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2SelectV2RWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/NotEqual:z:0lWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2:values:0MWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/Add:z:0*
T0	*#
_output_shapes
:?????????2P
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2kWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_row_lengths:0/RaggedFromRowSplits/control_dependency:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2

GatherV2?
@RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :2B
@RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/rank?
ARaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/ShapeShapeGatherV2:output:0*
T0	*
_output_shapes
:2C
ARaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/Shape?
jRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2l
jRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_type?
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*
_output_shapes
 2]
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok?
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack?
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1?
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2?
=RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_sliceStridedSliceGatherV2:output:0LRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2?
=RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice?
5RaggedFromRowSplits_3/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 27
5RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const?
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: 2F
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Equal?
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2E
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Rank?
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2L
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/start?
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2L
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/delta?
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: 2F
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range?
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: 2D
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All?
KRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2M
KRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const?
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2O
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1?
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = 2O
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2?
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = 2O
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3?
QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:08^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5003*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_50022S
QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard?
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2G
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack?
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1?
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2?
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1StridedSliceGatherV2:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2A
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1?
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack?
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2I
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1?
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2?
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2StridedSliceGatherV2:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2A
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2?
3RaggedFromRowSplits_3/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:?????????25
3RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub?
IRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 2K
IRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/Const?
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:?????????2a
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual?
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2]
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const?
YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: 2[
YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All?
bRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2d
bRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const?
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2f
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1?
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = 2f
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2?
hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branchwRu
sRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5039*
output_shapes
: *?
then_branchvRt
rRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_50382j
hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard?
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependencyIdentityGatherV2:output:0[^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*
_class
loc:@GatherV2*#
_output_shapes
:?????????2D
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency?
RaggedFromRowSplits_3/ShapeShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
:*
out_type0	2
RaggedFromRowSplits_3/Shape?
)RaggedFromRowSplits_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)RaggedFromRowSplits_3/strided_slice/stack?
+RaggedFromRowSplits_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+RaggedFromRowSplits_3/strided_slice/stack_1?
+RaggedFromRowSplits_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+RaggedFromRowSplits_3/strided_slice/stack_2?
#RaggedFromRowSplits_3/strided_sliceStridedSlice$RaggedFromRowSplits_3/Shape:output:02RaggedFromRowSplits_3/strided_slice/stack:output:04RaggedFromRowSplits_3/strided_slice/stack_1:output:04RaggedFromRowSplits_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2%
#RaggedFromRowSplits_3/strided_slice?
+RaggedFromRowSplits_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+RaggedFromRowSplits_3/strided_slice_1/stack?
-RaggedFromRowSplits_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-RaggedFromRowSplits_3/strided_slice_1/stack_1?
-RaggedFromRowSplits_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-RaggedFromRowSplits_3/strided_slice_1/stack_2?
%RaggedFromRowSplits_3/strided_slice_1StridedSliceKRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_3/strided_slice_1/stack:output:06RaggedFromRowSplits_3/strided_slice_1/stack_1:output:06RaggedFromRowSplits_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2'
%RaggedFromRowSplits_3/strided_slice_1?
*RaggedFromRowSplits_3/assert_equal_1/EqualEqual.RaggedFromRowSplits_3/strided_slice_1:output:0,RaggedFromRowSplits_3/strided_slice:output:0*
T0	*
_output_shapes
: 2,
*RaggedFromRowSplits_3/assert_equal_1/Equal?
)RaggedFromRowSplits_3/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2+
)RaggedFromRowSplits_3/assert_equal_1/Rank?
0RaggedFromRowSplits_3/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0RaggedFromRowSplits_3/assert_equal_1/range/start?
0RaggedFromRowSplits_3/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0RaggedFromRowSplits_3/assert_equal_1/range/delta?
*RaggedFromRowSplits_3/assert_equal_1/rangeRange9RaggedFromRowSplits_3/assert_equal_1/range/start:output:02RaggedFromRowSplits_3/assert_equal_1/Rank:output:09RaggedFromRowSplits_3/assert_equal_1/range/delta:output:0*
_output_shapes
: 2,
*RaggedFromRowSplits_3/assert_equal_1/range?
(RaggedFromRowSplits_3/assert_equal_1/AllAll.RaggedFromRowSplits_3/assert_equal_1/Equal:z:03RaggedFromRowSplits_3/assert_equal_1/range:output:0*
_output_shapes
: 2*
(RaggedFromRowSplits_3/assert_equal_1/All?
1RaggedFromRowSplits_3/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor23
1RaggedFromRowSplits_3/assert_equal_1/Assert/Const?
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:25
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_1?
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_3/strided_slice_1:0) = 25
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_2?
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_3/strided_slice:0) = 25
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_3?
7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_3/assert_equal_1/All:output:01RaggedFromRowSplits_3/assert_equal_1/All:output:0.RaggedFromRowSplits_3/strided_slice_1:output:0,RaggedFromRowSplits_3/strided_slice:output:0i^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_false_5076*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_true_507529
7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard?
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity?
/RaggedFromRowSplits_3/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :21
/RaggedFromRowSplits_3/assert_rank_at_least/rank?
0RaggedFromRowSplits_3/assert_rank_at_least/ShapeShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
:22
0RaggedFromRowSplits_3/assert_rank_at_least/Shape?
YRaggedFromRowSplits_3/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 2[
YRaggedFromRowSplits_3/assert_rank_at_least/assert_type/statically_determined_correct_type?
JRaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 2L
JRaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_ok?
(RaggedFromRowSplits_3/control_dependencyIdentityKRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_ok*
T0	*
_class
loc:@GatherV2*#
_output_shapes
:?????????2*
(RaggedFromRowSplits_3/control_dependency?
RaggedBoundingBox/ShapeShape1RaggedFromRowSplits_3/control_dependency:output:0*
T0	*
_output_shapes
:*
out_type0	2
RaggedBoundingBox/Shape?
RaggedBoundingBox/Shape_1ShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
:*
out_type0	2
RaggedBoundingBox/Shape_1?
%RaggedBoundingBox/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%RaggedBoundingBox/strided_slice/stack?
'RaggedBoundingBox/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedBoundingBox/strided_slice/stack_1?
'RaggedBoundingBox/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedBoundingBox/strided_slice/stack_2?
RaggedBoundingBox/strided_sliceStridedSlice RaggedBoundingBox/Shape:output:0.RaggedBoundingBox/strided_slice/stack:output:00RaggedBoundingBox/strided_slice/stack_1:output:00RaggedBoundingBox/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2!
RaggedBoundingBox/strided_slicet
RaggedBoundingBox/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
RaggedBoundingBox/sub/y?
RaggedBoundingBox/subSub(RaggedBoundingBox/strided_slice:output:0 RaggedBoundingBox/sub/y:output:0*
T0	*
_output_shapes
: 2
RaggedBoundingBox/sub?
'RaggedBoundingBox/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedBoundingBox/strided_slice_1/stack?
)RaggedBoundingBox/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)RaggedBoundingBox/strided_slice_1/stack_1?
)RaggedBoundingBox/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedBoundingBox/strided_slice_1/stack_2?
!RaggedBoundingBox/strided_slice_1StridedSlice1RaggedFromRowSplits_3/control_dependency:output:00RaggedBoundingBox/strided_slice_1/stack:output:02RaggedBoundingBox/strided_slice_1/stack_1:output:02RaggedBoundingBox/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2#
!RaggedBoundingBox/strided_slice_1?
'RaggedBoundingBox/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'RaggedBoundingBox/strided_slice_2/stack?
)RaggedBoundingBox/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)RaggedBoundingBox/strided_slice_2/stack_1?
)RaggedBoundingBox/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedBoundingBox/strided_slice_2/stack_2?
!RaggedBoundingBox/strided_slice_2StridedSlice1RaggedFromRowSplits_3/control_dependency:output:00RaggedBoundingBox/strided_slice_2/stack:output:02RaggedBoundingBox/strided_slice_2/stack_1:output:02RaggedBoundingBox/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2#
!RaggedBoundingBox/strided_slice_2?
RaggedBoundingBox/sub_1Sub*RaggedBoundingBox/strided_slice_1:output:0*RaggedBoundingBox/strided_slice_2:output:0*
T0	*#
_output_shapes
:?????????2
RaggedBoundingBox/sub_1|
RaggedBoundingBox/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
RaggedBoundingBox/Const?
RaggedBoundingBox/MaxMaxRaggedBoundingBox/sub_1:z:0 RaggedBoundingBox/Const:output:0*
T0	*
_output_shapes
: 2
RaggedBoundingBox/Max|
RaggedBoundingBox/Maximum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
RaggedBoundingBox/Maximum/y?
RaggedBoundingBox/MaximumMaximumRaggedBoundingBox/Max:output:0$RaggedBoundingBox/Maximum/y:output:0*
T0	*
_output_shapes
: 2
RaggedBoundingBox/Maximum?
'RaggedBoundingBox/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedBoundingBox/strided_slice_3/stack?
)RaggedBoundingBox/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)RaggedBoundingBox/strided_slice_3/stack_1?
)RaggedBoundingBox/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedBoundingBox/strided_slice_3/stack_2?
!RaggedBoundingBox/strided_slice_3StridedSlice"RaggedBoundingBox/Shape_1:output:00RaggedBoundingBox/strided_slice_3/stack:output:02RaggedBoundingBox/strided_slice_3/stack_1:output:02RaggedBoundingBox/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask2#
!RaggedBoundingBox/strided_slice_3?
RaggedBoundingBox/stackPackRaggedBoundingBox/sub:z:0RaggedBoundingBox/Maximum:z:0*
N*
T0	*
_output_shapes
:2
RaggedBoundingBox/stack?
RaggedBoundingBox/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
RaggedBoundingBox/concat/axis?
RaggedBoundingBox/concatConcatV2 RaggedBoundingBox/stack:output:0*RaggedBoundingBox/strided_slice_3:output:0&RaggedBoundingBox/concat/axis:output:0*
N*
T0	*
_output_shapes
:2
RaggedBoundingBox/concatt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSlice!RaggedBoundingBox/concat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
Fill/dims/1Const*
_output_shapes
: *
dtype0	*
value	B	 R2
Fill/dims/1z
	Fill/dimsPackstrided_slice:output:0Fill/dims/1:output:0*
N*
T0	*
_output_shapes
:2
	Fill/dimsx
FillFillFill/dims:output:0
fill_value*
T0	*'
_output_shapes
:?????????*

index_type0	2
Fill`
Fill_1/dims/1Const*
_output_shapes
: *
dtype0	*
value	B	 R2
Fill_1/dims/1?
Fill_1/dimsPackstrided_slice:output:0Fill_1/dims/1:output:0*
N*
T0	*
_output_shapes
:2
Fill_1/dims?
Fill_1FillFill_1/dims:output:0fill_1_value*
T0	*'
_output_shapes
:?????????*

index_type0	2
Fill_1?
#RaggedConcat/RaggedFromTensor/ShapeShapeFill:output:0*
T0	*
_output_shapes
:*
out_type0	2%
#RaggedConcat/RaggedFromTensor/Shape?
1RaggedConcat/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1RaggedConcat/RaggedFromTensor/strided_slice/stack?
3RaggedConcat/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3RaggedConcat/RaggedFromTensor/strided_slice/stack_1?
3RaggedConcat/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3RaggedConcat/RaggedFromTensor/strided_slice/stack_2?
+RaggedConcat/RaggedFromTensor/strided_sliceStridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0:RaggedConcat/RaggedFromTensor/strided_slice/stack:output:0<RaggedConcat/RaggedFromTensor/strided_slice/stack_1:output:0<RaggedConcat/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2-
+RaggedConcat/RaggedFromTensor/strided_slice?
3RaggedConcat/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3RaggedConcat/RaggedFromTensor/strided_slice_1/stack?
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_1?
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_2?
-RaggedConcat/RaggedFromTensor/strided_slice_1StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_1/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_1/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2/
-RaggedConcat/RaggedFromTensor/strided_slice_1?
3RaggedConcat/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3RaggedConcat/RaggedFromTensor/strided_slice_2/stack?
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_1?
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_2?
-RaggedConcat/RaggedFromTensor/strided_slice_2StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_2/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_2/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2/
-RaggedConcat/RaggedFromTensor/strided_slice_2?
!RaggedConcat/RaggedFromTensor/mulMul6RaggedConcat/RaggedFromTensor/strided_slice_1:output:06RaggedConcat/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: 2#
!RaggedConcat/RaggedFromTensor/mul?
3RaggedConcat/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3RaggedConcat/RaggedFromTensor/strided_slice_3/stack?
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_1?
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_2?
-RaggedConcat/RaggedFromTensor/strided_slice_3StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_3/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_3/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask2/
-RaggedConcat/RaggedFromTensor/strided_slice_3?
-RaggedConcat/RaggedFromTensor/concat/values_0Pack%RaggedConcat/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:2/
-RaggedConcat/RaggedFromTensor/concat/values_0?
)RaggedConcat/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)RaggedConcat/RaggedFromTensor/concat/axis?
$RaggedConcat/RaggedFromTensor/concatConcatV26RaggedConcat/RaggedFromTensor/concat/values_0:output:06RaggedConcat/RaggedFromTensor/strided_slice_3:output:02RaggedConcat/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:2&
$RaggedConcat/RaggedFromTensor/concat?
%RaggedConcat/RaggedFromTensor/ReshapeReshapeFill:output:0-RaggedConcat/RaggedFromTensor/concat:output:0*
T0	*
Tshape0	*#
_output_shapes
:?????????2'
%RaggedConcat/RaggedFromTensor/Reshape?
3RaggedConcat/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3RaggedConcat/RaggedFromTensor/strided_slice_4/stack?
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_1?
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_2?
-RaggedConcat/RaggedFromTensor/strided_slice_4StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_4/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_4/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2/
-RaggedConcat/RaggedFromTensor/strided_slice_4?
#RaggedConcat/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R2%
#RaggedConcat/RaggedFromTensor/Const?
>RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape.RaggedConcat/RaggedFromTensor/Reshape:output:0*
T0	*
_output_shapes
:*
out_type0	2@
>RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/Shape?
LRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2N
LRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack?
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2P
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1?
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2P
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2?
FRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSliceGRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0URaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0WRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0WRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2H
FRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice?
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2a
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y?
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV26RaggedConcat/RaggedFromTensor/strided_slice_4:output:0hRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: 2_
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add?
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2g
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start?
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2g
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta?
dRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/CastCastnRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2f
dRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast?
fRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1CastnRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2h
fRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1?
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangehRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast:y:0aRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0jRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2a
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range?
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMulhRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0,RaggedConcat/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:?????????2_
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul?
%RaggedConcat/RaggedFromTensor_1/ShapeShapeFill_1:output:0*
T0	*
_output_shapes
:*
out_type0	2'
%RaggedConcat/RaggedFromTensor_1/Shape?
3RaggedConcat/RaggedFromTensor_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3RaggedConcat/RaggedFromTensor_1/strided_slice/stack?
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_1?
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_2?
-RaggedConcat/RaggedFromTensor_1/strided_sliceStridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0<RaggedConcat/RaggedFromTensor_1/strided_slice/stack:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice/stack_1:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2/
-RaggedConcat/RaggedFromTensor_1/strided_slice?
5RaggedConcat/RaggedFromTensor_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack?
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_1?
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_2?
/RaggedConcat/RaggedFromTensor_1/strided_slice_1StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask21
/RaggedConcat/RaggedFromTensor_1/strided_slice_1?
5RaggedConcat/RaggedFromTensor_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack?
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_1?
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_2?
/RaggedConcat/RaggedFromTensor_1/strided_slice_2StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask21
/RaggedConcat/RaggedFromTensor_1/strided_slice_2?
#RaggedConcat/RaggedFromTensor_1/mulMul8RaggedConcat/RaggedFromTensor_1/strided_slice_1:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_2:output:0*
T0	*
_output_shapes
: 2%
#RaggedConcat/RaggedFromTensor_1/mul?
5RaggedConcat/RaggedFromTensor_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack?
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_1?
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_2?
/RaggedConcat/RaggedFromTensor_1/strided_slice_3StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask21
/RaggedConcat/RaggedFromTensor_1/strided_slice_3?
/RaggedConcat/RaggedFromTensor_1/concat/values_0Pack'RaggedConcat/RaggedFromTensor_1/mul:z:0*
N*
T0	*
_output_shapes
:21
/RaggedConcat/RaggedFromTensor_1/concat/values_0?
+RaggedConcat/RaggedFromTensor_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+RaggedConcat/RaggedFromTensor_1/concat/axis?
&RaggedConcat/RaggedFromTensor_1/concatConcatV28RaggedConcat/RaggedFromTensor_1/concat/values_0:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_3:output:04RaggedConcat/RaggedFromTensor_1/concat/axis:output:0*
N*
T0	*
_output_shapes
:2(
&RaggedConcat/RaggedFromTensor_1/concat?
'RaggedConcat/RaggedFromTensor_1/ReshapeReshapeFill_1:output:0/RaggedConcat/RaggedFromTensor_1/concat:output:0*
T0	*
Tshape0	*#
_output_shapes
:?????????2)
'RaggedConcat/RaggedFromTensor_1/Reshape?
5RaggedConcat/RaggedFromTensor_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack?
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_1?
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_2?
/RaggedConcat/RaggedFromTensor_1/strided_slice_4StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask21
/RaggedConcat/RaggedFromTensor_1/strided_slice_4?
%RaggedConcat/RaggedFromTensor_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R2'
%RaggedConcat/RaggedFromTensor_1/Const?
@RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/ShapeShape0RaggedConcat/RaggedFromTensor_1/Reshape:output:0*
T0	*
_output_shapes
:*
out_type0	2B
@RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/Shape?
NRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
NRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack?
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_1?
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_2?
HRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_sliceStridedSliceIRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/Shape:output:0WRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack:output:0YRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_1:output:0YRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2J
HRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice?
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2c
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y?
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV28RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:0jRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: 2a
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add?
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2i
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start?
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2i
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta?
fRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/CastCastpRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2h
fRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast?
hRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1CastpRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2j
hRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1?
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangejRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast:y:0cRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0lRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2c
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range?
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMuljRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0.RaggedConcat/RaggedFromTensor_1/Const:output:0*
T0	*#
_output_shapes
:?????????2a
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul?
 RaggedConcat/RaggedNRows_1/ShapeShape1RaggedFromRowSplits_3/control_dependency:output:0*
T0	*
_output_shapes
:*
out_type0	2"
 RaggedConcat/RaggedNRows_1/Shape?
.RaggedConcat/RaggedNRows_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.RaggedConcat/RaggedNRows_1/strided_slice/stack?
0RaggedConcat/RaggedNRows_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0RaggedConcat/RaggedNRows_1/strided_slice/stack_1?
0RaggedConcat/RaggedNRows_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0RaggedConcat/RaggedNRows_1/strided_slice/stack_2?
(RaggedConcat/RaggedNRows_1/strided_sliceStridedSlice)RaggedConcat/RaggedNRows_1/Shape:output:07RaggedConcat/RaggedNRows_1/strided_slice/stack:output:09RaggedConcat/RaggedNRows_1/strided_slice/stack_1:output:09RaggedConcat/RaggedNRows_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2*
(RaggedConcat/RaggedNRows_1/strided_slice?
 RaggedConcat/RaggedNRows_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 RaggedConcat/RaggedNRows_1/sub/y?
RaggedConcat/RaggedNRows_1/subSub1RaggedConcat/RaggedNRows_1/strided_slice:output:0)RaggedConcat/RaggedNRows_1/sub/y:output:0*
T0	*
_output_shapes
: 2 
RaggedConcat/RaggedNRows_1/sub?
!RaggedConcat/assert_equal_1/EqualEqual"RaggedConcat/RaggedNRows_1/sub:z:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:0*
T0	*
_output_shapes
: 2#
!RaggedConcat/assert_equal_1/Equal?
 RaggedConcat/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : 2"
 RaggedConcat/assert_equal_1/Rank?
'RaggedConcat/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'RaggedConcat/assert_equal_1/range/start?
'RaggedConcat/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'RaggedConcat/assert_equal_1/range/delta?
!RaggedConcat/assert_equal_1/rangeRange0RaggedConcat/assert_equal_1/range/start:output:0)RaggedConcat/assert_equal_1/Rank:output:00RaggedConcat/assert_equal_1/range/delta:output:0*
_output_shapes
: 2#
!RaggedConcat/assert_equal_1/range?
RaggedConcat/assert_equal_1/AllAll%RaggedConcat/assert_equal_1/Equal:z:0*RaggedConcat/assert_equal_1/range:output:0*
_output_shapes
: 2!
RaggedConcat/assert_equal_1/All?
(RaggedConcat/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'Input tensors have incompatible shapes.2*
(RaggedConcat/assert_equal_1/Assert/Const?
*RaggedConcat/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2,
*RaggedConcat/assert_equal_1/Assert/Const_1?
*RaggedConcat/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*8
value/B- B'x (RaggedConcat/RaggedNRows_1/sub:0) = 2,
*RaggedConcat/assert_equal_1/Assert/Const_2?
*RaggedConcat/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = 2,
*RaggedConcat/assert_equal_1/Assert/Const_3?
.RaggedConcat/assert_equal_1/Assert/AssertGuardIf(RaggedConcat/assert_equal_1/All:output:0(RaggedConcat/assert_equal_1/All:output:0"RaggedConcat/RaggedNRows_1/sub:z:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:08^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *L
else_branch=R;
9RaggedConcat_assert_equal_1_Assert_AssertGuard_false_5236*
output_shapes
: *K
then_branch<R:
8RaggedConcat_assert_equal_1_Assert_AssertGuard_true_523520
.RaggedConcat/assert_equal_1/Assert/AssertGuard?
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentity7RaggedConcat/assert_equal_1/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 29
7RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity?
!RaggedConcat/assert_equal_3/EqualEqual8RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:0*
T0	*
_output_shapes
: 2#
!RaggedConcat/assert_equal_3/Equal?
 RaggedConcat/assert_equal_3/RankConst*
_output_shapes
: *
dtype0*
value	B : 2"
 RaggedConcat/assert_equal_3/Rank?
'RaggedConcat/assert_equal_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'RaggedConcat/assert_equal_3/range/start?
'RaggedConcat/assert_equal_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'RaggedConcat/assert_equal_3/range/delta?
!RaggedConcat/assert_equal_3/rangeRange0RaggedConcat/assert_equal_3/range/start:output:0)RaggedConcat/assert_equal_3/Rank:output:00RaggedConcat/assert_equal_3/range/delta:output:0*
_output_shapes
: 2#
!RaggedConcat/assert_equal_3/range?
RaggedConcat/assert_equal_3/AllAll%RaggedConcat/assert_equal_3/Equal:z:0*RaggedConcat/assert_equal_3/range:output:0*
_output_shapes
: 2!
RaggedConcat/assert_equal_3/All?
(RaggedConcat/assert_equal_3/Assert/ConstConst*
_output_shapes
: *
dtype0*8
value/B- B'Input tensors have incompatible shapes.2*
(RaggedConcat/assert_equal_3/Assert/Const?
*RaggedConcat/assert_equal_3/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2,
*RaggedConcat/assert_equal_3/Assert/Const_1?
*RaggedConcat/assert_equal_3/Assert/Const_2Const*
_output_shapes
: *
dtype0*I
value@B> B8x (RaggedConcat/RaggedFromTensor_1/strided_slice_4:0) = 2,
*RaggedConcat/assert_equal_3/Assert/Const_2?
*RaggedConcat/assert_equal_3/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = 2,
*RaggedConcat/assert_equal_3/Assert/Const_3?
.RaggedConcat/assert_equal_3/Assert/AssertGuardIf(RaggedConcat/assert_equal_3/All:output:0(RaggedConcat/assert_equal_3/All:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:0/^RaggedConcat/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *L
else_branch=R;
9RaggedConcat_assert_equal_3_Assert_AssertGuard_false_5266*
output_shapes
: *K
then_branch<R:
8RaggedConcat_assert_equal_3_Assert_AssertGuard_true_526520
.RaggedConcat/assert_equal_3/Assert/AssertGuard?
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentity7RaggedConcat/assert_equal_3/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 29
7RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity?
RaggedConcat/concat/axisConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
RaggedConcat/concat/axis?
RaggedConcat/concatConcatV2.RaggedConcat/RaggedFromTensor/Reshape:output:0WWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:00RaggedConcat/RaggedFromTensor_1/Reshape:output:0!RaggedConcat/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2
RaggedConcat/concat?
 RaggedConcat/strided_slice/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 RaggedConcat/strided_slice/stack?
"RaggedConcat/strided_slice/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: 2$
"RaggedConcat/strided_slice/stack_1?
"RaggedConcat/strided_slice/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2$
"RaggedConcat/strided_slice/stack_2?
RaggedConcat/strided_sliceStridedSliceaRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0)RaggedConcat/strided_slice/stack:output:0+RaggedConcat/strided_slice/stack_1:output:0+RaggedConcat/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
RaggedConcat/strided_slice?
"RaggedConcat/strided_slice_1/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2$
"RaggedConcat/strided_slice_1/stack?
$RaggedConcat/strided_slice_1/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: 2&
$RaggedConcat/strided_slice_1/stack_1?
$RaggedConcat/strided_slice_1/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2&
$RaggedConcat/strided_slice_1/stack_2?
RaggedConcat/strided_slice_1StridedSlice1RaggedFromRowSplits_3/control_dependency:output:0+RaggedConcat/strided_slice_1/stack:output:0-RaggedConcat/strided_slice_1/stack_1:output:0-RaggedConcat/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2
RaggedConcat/strided_slice_1?
RaggedConcat/addAddV2%RaggedConcat/strided_slice_1:output:0#RaggedConcat/strided_slice:output:0*
T0	*#
_output_shapes
:?????????2
RaggedConcat/add?
"RaggedConcat/strided_slice_2/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"RaggedConcat/strided_slice_2/stack?
$RaggedConcat/strided_slice_2/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: 2&
$RaggedConcat/strided_slice_2/stack_1?
$RaggedConcat/strided_slice_2/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2&
$RaggedConcat/strided_slice_2/stack_2?
RaggedConcat/strided_slice_2StridedSlice1RaggedFromRowSplits_3/control_dependency:output:0+RaggedConcat/strided_slice_2/stack:output:0-RaggedConcat/strided_slice_2/stack_1:output:0-RaggedConcat/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
RaggedConcat/strided_slice_2?
RaggedConcat/add_1AddV2#RaggedConcat/strided_slice:output:0%RaggedConcat/strided_slice_2:output:0*
T0	*
_output_shapes
: 2
RaggedConcat/add_1?
"RaggedConcat/strided_slice_3/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2$
"RaggedConcat/strided_slice_3/stack?
$RaggedConcat/strided_slice_3/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: 2&
$RaggedConcat/strided_slice_3/stack_1?
$RaggedConcat/strided_slice_3/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2&
$RaggedConcat/strided_slice_3/stack_2?
RaggedConcat/strided_slice_3StridedSlicecRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0+RaggedConcat/strided_slice_3/stack:output:0-RaggedConcat/strided_slice_3/stack_1:output:0-RaggedConcat/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2
RaggedConcat/strided_slice_3?
RaggedConcat/add_2AddV2%RaggedConcat/strided_slice_3:output:0RaggedConcat/add_1:z:0*
T0	*#
_output_shapes
:?????????2
RaggedConcat/add_2?
"RaggedConcat/strided_slice_4/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"RaggedConcat/strided_slice_4/stack?
$RaggedConcat/strided_slice_4/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: 2&
$RaggedConcat/strided_slice_4/stack_1?
$RaggedConcat/strided_slice_4/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2&
$RaggedConcat/strided_slice_4/stack_2?
RaggedConcat/strided_slice_4StridedSlicecRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0+RaggedConcat/strided_slice_4/stack:output:0-RaggedConcat/strided_slice_4/stack_1:output:0-RaggedConcat/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
RaggedConcat/strided_slice_4?
RaggedConcat/add_3AddV2RaggedConcat/add_1:z:0%RaggedConcat/strided_slice_4:output:0*
T0	*
_output_shapes
: 2
RaggedConcat/add_3?
RaggedConcat/concat_1/axisConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
RaggedConcat/concat_1/axis?
RaggedConcat/concat_1ConcatV2aRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0RaggedConcat/add:z:0RaggedConcat/add_2:z:0#RaggedConcat/concat_1/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2
RaggedConcat/concat_1?
RaggedConcat/mul/yConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R2
RaggedConcat/mul/y?
RaggedConcat/mulMul6RaggedConcat/RaggedFromTensor/strided_slice_4:output:0RaggedConcat/mul/y:output:0*
T0	*
_output_shapes
: 2
RaggedConcat/mul?
RaggedConcat/range/startConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
RaggedConcat/range/start?
RaggedConcat/range/deltaConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :2
RaggedConcat/range/delta?
RaggedConcat/range/CastCast!RaggedConcat/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
RaggedConcat/range/Cast?
RaggedConcat/range/Cast_1Cast!RaggedConcat/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
RaggedConcat/range/Cast_1?
RaggedConcat/rangeRangeRaggedConcat/range/Cast:y:0RaggedConcat/mul:z:0RaggedConcat/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2
RaggedConcat/range?
RaggedConcat/Reshape/shapeConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"   ????2
RaggedConcat/Reshape/shape?
RaggedConcat/ReshapeReshapeRaggedConcat/range:output:0#RaggedConcat/Reshape/shape:output:0*
T0	*'
_output_shapes
:?????????2
RaggedConcat/Reshape?
RaggedConcat/transpose/permConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       2
RaggedConcat/transpose/perm?
RaggedConcat/transpose	TransposeRaggedConcat/Reshape:output:0$RaggedConcat/transpose/perm:output:0*
T0	*'
_output_shapes
:?????????2
RaggedConcat/transpose?
RaggedConcat/Reshape_1/shapeConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
?????????2
RaggedConcat/Reshape_1/shape?
RaggedConcat/Reshape_1ReshapeRaggedConcat/transpose:y:0%RaggedConcat/Reshape_1/shape:output:0*
T0	*#
_output_shapes
:?????????2
RaggedConcat/Reshape_1?
&RaggedConcat/RaggedGather/RaggedGatherRaggedGatherRaggedConcat/concat_1:output:0RaggedConcat/concat:output:0RaggedConcat/Reshape_1:output:0*
OUTPUT_RAGGED_RANK*
PARAMS_RAGGED_RANK*
Tindices0	*
Tvalues0	*2
_output_shapes 
:?????????:?????????2(
&RaggedConcat/RaggedGather/RaggedGather?
"RaggedConcat/strided_slice_5/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: 2$
"RaggedConcat/strided_slice_5/stack?
$RaggedConcat/strided_slice_5/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: 2&
$RaggedConcat/strided_slice_5/stack_1?
$RaggedConcat/strided_slice_5/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:2&
$RaggedConcat/strided_slice_5/stack_2?
RaggedConcat/strided_slice_5StridedSlice=RaggedConcat/RaggedGather/RaggedGather:output_nested_splits:0+RaggedConcat/strided_slice_5/stack:output:0-RaggedConcat/strided_slice_5/stack_1:output:0-RaggedConcat/strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask2
RaggedConcat/strided_slice_5?
IdentityIdentity<RaggedConcat/RaggedGather/RaggedGather:output_dense_values:0/^RaggedConcat/assert_equal_1/Assert/AssertGuard/^RaggedConcat/assert_equal_3/Assert/AssertGuardP^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardg^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard6^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardd^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2b^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2W^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets*
T0	*#
_output_shapes
:?????????2

Identity?

Identity_1Identity%RaggedConcat/strided_slice_5:output:0/^RaggedConcat/assert_equal_1/Assert/AssertGuard/^RaggedConcat/assert_equal_3/Assert/AssertGuardP^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardg^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard6^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardd^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2b^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2W^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets*
T0	*#
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : : : 2`
.RaggedConcat/assert_equal_1/Assert/AssertGuard.RaggedConcat/assert_equal_1/Assert/AssertGuard2`
.RaggedConcat/assert_equal_3/Assert/AssertGuard.RaggedConcat/assert_equal_3/Assert/AssertGuard2?
ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2?
fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2n
5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard2?
QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2?
hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard2?
QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2?
hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard2?
QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2?
hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard2?
cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV22?
aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV22?
VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsVWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:L H
#
_output_shapes
:?????????
!
_user_specified_name	strings:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
rRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4798?
?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
?
mRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2o
mRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp?
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2u
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?
?
[RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5002?
?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
?
VRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2X
VRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp?
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2^
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
+
__inference__destroyer_5351
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
__inference_detokenize_4579
	tokenized	
tokenized_1	0
,none_export_lookuptableexportv2_table_handle
identity??Assert/Assert?None_Export/LookupTableExportV2?
None_Export/LookupTableExportV2LookupTableExportV2,none_export_lookuptableexportv2_table_handle*
Tkeys0*
Tvalues0	*
_output_shapes

::2!
None_Export/LookupTableExportV2?
EnsureShapeEnsureShape&None_Export/LookupTableExportV2:keys:0*
T0*#
_output_shapes
:?????????*
shape:?????????2
EnsureShape?
EnsureShape_1EnsureShape(None_Export/LookupTableExportV2:values:0*
T0	*#
_output_shapes
:?????????*
shape:?????????2
EnsureShape_1g
argsort/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
argsort/axisg
argsort/NegNegEnsureShape_1:output:0*
T0	*#
_output_shapes
:?????????2
argsort/Neg]
argsort/ShapeShapeargsort/Neg:y:0*
T0	*
_output_shapes
:2
argsort/Shape?
argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
argsort/strided_slice/stack?
argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
argsort/strided_slice/stack_1?
argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
argsort/strided_slice/stack_2?
argsort/strided_sliceStridedSliceargsort/Shape:output:0$argsort/strided_slice/stack:output:0&argsort/strided_slice/stack_1:output:0&argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
argsort/strided_slice^
argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2
argsort/Rank?
argsort/TopKV2TopKV2argsort/Neg:y:0argsort/strided_slice:output:0*
T0	*2
_output_shapes 
:?????????:?????????2
argsort/TopKV2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2EnsureShape_1:output:0argsort/TopKV2:indices:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:?????????2

GatherV2d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2EnsureShape:output:0argsort/TopKV2:indices:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceGatherV2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yb
EqualEqualstrided_slice:output:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equalx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceGatherV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceGatherV2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2
strided_slice_2s
subSubstrided_slice_1:output:0strided_slice_2:output:0*
T0	*#
_output_shapes
:?????????2
subX
	Equal_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
	Equal_1/yf
Equal_1Equalsub:z:0Equal_1/y:output:0*
T0	*#
_output_shapes
:?????????2	
Equal_1X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstF
AllAllEqual_1:z:0Const:output:0*
_output_shapes
: 2
AllI
and
LogicalAnd	Equal:z:0All:output:0*
_output_shapes
: 2
and?
Assert/ConstConst*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`2
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`2
Assert/Assert/data_0r
Assert/AssertAssertand:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/AssertZ
SizeSizeGatherV2_1:output:0^Assert/Assert*
T0*
_output_shapes
: 2
SizeS
CastCastSize:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Castd
	Minimum_1Minimum	tokenizedCast:y:0*
T0	*#
_output_shapes
:?????????2
	Minimum_1p
concat/values_1Const*
_output_shapes
:*
dtype0*
valueBB[UNK]2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2GatherV2_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:?????????2
concatz
RaggedGather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
RaggedGather/GatherV2/axis?
RaggedGather/GatherV2GatherV2concat:output:0Minimum_1:z:0#RaggedGather/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
RaggedGather/GatherV2m
RaggedSegmentJoin/ShapeShapetokenized_1*
T0	*
_output_shapes
:2
RaggedSegmentJoin/Shape?
%RaggedSegmentJoin/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%RaggedSegmentJoin/strided_slice/stack?
'RaggedSegmentJoin/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedSegmentJoin/strided_slice/stack_1?
'RaggedSegmentJoin/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedSegmentJoin/strided_slice/stack_2?
RaggedSegmentJoin/strided_sliceStridedSlice RaggedSegmentJoin/Shape:output:0.RaggedSegmentJoin/strided_slice/stack:output:00RaggedSegmentJoin/strided_slice/stack_1:output:00RaggedSegmentJoin/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
RaggedSegmentJoin/strided_slicet
RaggedSegmentJoin/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
RaggedSegmentJoin/sub/y?
RaggedSegmentJoin/subSub(RaggedSegmentJoin/strided_slice:output:0 RaggedSegmentJoin/sub/y:output:0*
T0*
_output_shapes
: 2
RaggedSegmentJoin/sub?
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2?
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_sliceStridedSlicetokenized_1GRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2:
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1StridedSlicetokenized_1IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2<
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1?
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/subSubARaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice:output:0CRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????20
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub?
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/ShapeShapetokenized_1*
T0	*
_output_shapes
:*
out_type0	22
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2<
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2?
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y?
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1SubCRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2:output:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 22
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta?
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/CastCast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 27
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast?
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1Cast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 29
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1?
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/rangeRange9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast:y:04RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1:z:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????22
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/range?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/CastCast2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast?
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ShapeShape9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:29
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape?
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlice@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackHRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo:RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast:y:0LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2?
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 29
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const?
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaxMaxFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 27
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max?
;RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2=
;RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumDRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0>RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2;
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2O
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2K
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastRRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2>
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackKRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/TileTileERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2T
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0[RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2N
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2W
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdURaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0^RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0SRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2G
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeGRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2J
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereQRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2I
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
%RaggedSegmentJoin/UnsortedSegmentJoinUnsortedSegmentJoinRaggedGather/GatherV2:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin/sub:z:0*
Tindices0	*#
_output_shapes
:?????????*
	separator 2'
%RaggedSegmentJoin/UnsortedSegmentJoin?
StaticRegexReplaceStaticRegexReplace.RaggedSegmentJoin/UnsortedSegmentJoin:output:0*#
_output_shapes
:?????????*
pattern \#\#*
rewrite 2
StaticRegexReplace?
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:?????????*
pattern	^ +| +$*
rewrite 2
StaticRegexReplace_1h
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
value	B B 2
StringSplit/Const?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_1:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2
StringSplit/StringSplitV2?
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
StringSplit/strided_slice/stack?
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!StringSplit/strided_slice/stack_1?
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!StringSplit/strided_slice/stack_2?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
StringSplit/strided_slice?
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!StringSplit/strided_slice_1/stack?
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_1?
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_2?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
StringSplit/strided_slice_1?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2D
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2M
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2R
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2M
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2Q
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2O
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
StaticRegexFullMatch_1StaticRegexFullMatch"StringSplit/StringSplitV2:values:0*#
_output_shapes
:?????????*&
pattern\[PAD\]|\[START\]|\[END\]2
StaticRegexFullMatch_1p
LogicalNot_1
LogicalNotStaticRegexFullMatch_1:output:0*#
_output_shapes
:?????????2
LogicalNot_1Z
RaggedMask/assert_equal/NoOpNoOp*
_output_shapes
 2
RaggedMask/assert_equal/NoOp?
RaggedMask/CastCastLogicalNot_1:y:0^RaggedMask/assert_equal/NoOp*

DstT0	*

SrcT0
*#
_output_shapes
:?????????2
RaggedMask/Cast?
 RaggedMask/RaggedReduceSum/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:2"
 RaggedMask/RaggedReduceSum/Shape?
.RaggedMask/RaggedReduceSum/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 20
.RaggedMask/RaggedReduceSum/strided_slice/stack?
0RaggedMask/RaggedReduceSum/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:22
0RaggedMask/RaggedReduceSum/strided_slice/stack_1?
0RaggedMask/RaggedReduceSum/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:22
0RaggedMask/RaggedReduceSum/strided_slice/stack_2?
(RaggedMask/RaggedReduceSum/strided_sliceStridedSlice)RaggedMask/RaggedReduceSum/Shape:output:07RaggedMask/RaggedReduceSum/strided_slice/stack:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_1:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(RaggedMask/RaggedReduceSum/strided_slice?
 RaggedMask/RaggedReduceSum/sub/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2"
 RaggedMask/RaggedReduceSum/sub/y?
RaggedMask/RaggedReduceSum/subSub1RaggedMask/RaggedReduceSum/strided_slice:output:0)RaggedMask/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: 2 
RaggedMask/RaggedReduceSum/sub?
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2?
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2C
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2E
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1?
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubJRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????29
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub?
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:*
out_type0	2;
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2E
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2?
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0	*
value	B	 R2=
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y?
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubLRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2;
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta?
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2@
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast?
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2B
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1?
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0=RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2;
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCast;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast?
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2B
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape?
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2P
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2R
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2R
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceIRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackQRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToCRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2H
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2B
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const?
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2@
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max?
DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2F
DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumMRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2D
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
?????????2X
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2T
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2G
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackTRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileNRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2]
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0dRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2W
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2`
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0gRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2U
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0\RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2P
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapePRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2S
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereZRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2W
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2R
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
-RaggedMask/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumRaggedMask/Cast:y:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0"RaggedMask/RaggedReduceSum/sub:z:0*
T0	*
Tindices0	*#
_output_shapes
:?????????2/
-RaggedMask/RaggedReduceSum/UnsortedSegmentSum?
RaggedMask/Cumsum/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2
RaggedMask/Cumsum/axis?
RaggedMask/CumsumCumsum6RaggedMask/RaggedReduceSum/UnsortedSegmentSum:output:0RaggedMask/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2
RaggedMask/Cumsum?
RaggedMask/concat/values_0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0	*
valueB	R 2
RaggedMask/concat/values_0?
RaggedMask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
?????????2
RaggedMask/concat/axis?
RaggedMask/concatConcatV2#RaggedMask/concat/values_0:output:0RaggedMask/Cumsum:out:0RaggedMask/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2
RaggedMask/concat?
(RaggedMask/RaggedMask/boolean_mask/ShapeShape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
:2*
(RaggedMask/RaggedMask/boolean_mask/Shape?
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 28
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stack?
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1?
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2?
0RaggedMask/RaggedMask/boolean_mask/strided_sliceStridedSlice1RaggedMask/RaggedMask/boolean_mask/Shape:output:0?RaggedMask/RaggedMask/boolean_mask/strided_slice/stack:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:22
0RaggedMask/RaggedMask/boolean_mask/strided_slice?
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2;
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices?
'RaggedMask/RaggedMask/boolean_mask/ProdProd9RaggedMask/RaggedMask/boolean_mask/strided_slice:output:0BRaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2)
'RaggedMask/RaggedMask/boolean_mask/Prod?
*RaggedMask/RaggedMask/boolean_mask/Shape_1Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
:2,
*RaggedMask/RaggedMask/boolean_mask/Shape_1?
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2?
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask24
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1?
*RaggedMask/RaggedMask/boolean_mask/Shape_2Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
:2,
*RaggedMask/RaggedMask/boolean_mask/Shape_2?
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2?
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_2:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask24
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2?
2RaggedMask/RaggedMask/boolean_mask/concat/values_1Pack0RaggedMask/RaggedMask/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:24
2RaggedMask/RaggedMask/boolean_mask/concat/values_1?
.RaggedMask/RaggedMask/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 20
.RaggedMask/RaggedMask/boolean_mask/concat/axis?
)RaggedMask/RaggedMask/boolean_mask/concatConcatV2;RaggedMask/RaggedMask/boolean_mask/strided_slice_1:output:0;RaggedMask/RaggedMask/boolean_mask/concat/values_1:output:0;RaggedMask/RaggedMask/boolean_mask/strided_slice_2:output:07RaggedMask/RaggedMask/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)RaggedMask/RaggedMask/boolean_mask/concat?
*RaggedMask/RaggedMask/boolean_mask/ReshapeReshape"StringSplit/StringSplitV2:values:02RaggedMask/RaggedMask/boolean_mask/concat:output:0*
T0*#
_output_shapes
:?????????2,
*RaggedMask/RaggedMask/boolean_mask/Reshape?
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????24
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape?
,RaggedMask/RaggedMask/boolean_mask/Reshape_1ReshapeLogicalNot_1:y:0;RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2.
,RaggedMask/RaggedMask/boolean_mask/Reshape_1?
(RaggedMask/RaggedMask/boolean_mask/WhereWhere5RaggedMask/RaggedMask/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2*
(RaggedMask/RaggedMask/boolean_mask/Where?
*RaggedMask/RaggedMask/boolean_mask/SqueezeSqueeze0RaggedMask/RaggedMask/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2,
*RaggedMask/RaggedMask/boolean_mask/Squeeze?
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 22
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axis?
+RaggedMask/RaggedMask/boolean_mask/GatherV2GatherV23RaggedMask/RaggedMask/boolean_mask/Reshape:output:03RaggedMask/RaggedMask/boolean_mask/Squeeze:output:09RaggedMask/RaggedMask/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2-
+RaggedMask/RaggedMask/boolean_mask/GatherV2?
RaggedSegmentJoin_1/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:2
RaggedSegmentJoin_1/Shape?
'RaggedSegmentJoin_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'RaggedSegmentJoin_1/strided_slice/stack?
)RaggedSegmentJoin_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedSegmentJoin_1/strided_slice/stack_1?
)RaggedSegmentJoin_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedSegmentJoin_1/strided_slice/stack_2?
!RaggedSegmentJoin_1/strided_sliceStridedSlice"RaggedSegmentJoin_1/Shape:output:00RaggedSegmentJoin_1/strided_slice/stack:output:02RaggedSegmentJoin_1/strided_slice/stack_1:output:02RaggedSegmentJoin_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!RaggedSegmentJoin_1/strided_slicex
RaggedSegmentJoin_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
RaggedSegmentJoin_1/sub/y?
RaggedSegmentJoin_1/subSub*RaggedSegmentJoin_1/strided_slice:output:0"RaggedSegmentJoin_1/sub/y:output:0*
T0*
_output_shapes
: 2
RaggedSegmentJoin_1/sub?
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2?
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceRaggedMask/concat:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2<
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceRaggedMask/concat:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2>
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1?
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/subSubCRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice:output:0ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????22
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub?
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:*
out_type0	24
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2>
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2?
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R26
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y?
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1SubERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 24
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta?
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/CastCastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 29
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast?
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1CastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2;
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1?
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/rangeRange;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast:y:06RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1:z:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????24
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/CastCast4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast?
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ShapeShape;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2;
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape?
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceBRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackJRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2A
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const?
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 29
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max?
=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2?
=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumFRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2=
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2Q
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2M
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastTRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2@
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackMRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/TileTileGRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0]RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2P
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2Y
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdWRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0`RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0URaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2I
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeIRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2L
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereSRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2K
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
'RaggedSegmentJoin_1/UnsortedSegmentJoinUnsortedSegmentJoin4RaggedMask/RaggedMask/boolean_mask/GatherV2:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin_1/sub:z:0*
Tindices0	*#
_output_shapes
:?????????*
	separator 2)
'RaggedSegmentJoin_1/UnsortedSegmentJoin?
IdentityIdentity0RaggedSegmentJoin_1/UnsortedSegmentJoin:output:0^Assert/Assert ^None_Export/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????:?????????: 2
Assert/AssertAssert/Assert2B
None_Export/LookupTableExportV2None_Export/LookupTableExportV2:N J
#
_output_shapes
:?????????
#
_user_specified_name	tokenized:NJ
#
_output_shapes
:?????????
#
_user_specified_name	tokenized
?
?
ARaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_true_5075m
iraggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_assert_equal_1_all
G
Craggedfromrowsplits_3_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_3_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1
?
<RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2>
<RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOp?
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_assert_equal_1_all=^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2D
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1"?
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
 __inference__traced_restore_5409
file_prefix+
assignvariableop_variable_1:
??

identity_2??AssignVariableOp?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#en/vocab/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variable_1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp9
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp{

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_1m

Identity_2IdentityIdentity_1:output:0^AssignVariableOp*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*
_input_shapes
: : 2$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
9RaggedConcat_assert_equal_1_Assert_AssertGuard_false_5236Y
Uraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_all
X
Traggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggednrows_1_sub	g
craggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4	=
9raggedconcat_assert_equal_1_assert_assertguard_identity_1
??5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert?
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*8
value/B- B'Input tensors have incompatible shapes.2>
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_0?
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2>
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_1?
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*8
value/B- B'x (RaggedConcat/RaggedNRows_1/sub:0) = 2>
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_2?
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = 2>
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_4?
5RaggedConcat/assert_equal_1/Assert/AssertGuard/AssertAssertUraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_allERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Traggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggednrows_1_subERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0craggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4*
T

2		*
_output_shapes
 27
5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert?
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentityUraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_all6^RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 29
7RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity?
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity:output:06^RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2;
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1"
9raggedconcat_assert_equal_1_assert_assertguard_identity_1BRaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2n
5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
pRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4685?
?raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
v
rraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	u
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
?
kRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2m
kRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp?
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alll^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2q
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityxRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1zRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?
?
__inference__traced_save_5396
file_prefix)
%savev2_variable_1_read_readvariableop
savev2_const_3

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#en/vocab/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_variable_1_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :??: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:"

_output_shapes

:??:

_output_shapes
: 
?

?
__inference_lookup_4615
	token_ids	
token_ids_1	
gather_resource:
??
identity

identity_1	??RaggedGather/ReadVariableOp?
RaggedGather/ReadVariableOpReadVariableOpgather_resource*
_output_shapes

:??*
dtype02
RaggedGather/ReadVariableOpz
RaggedGather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
RaggedGather/GatherV2/axis?
RaggedGather/GatherV2GatherV2#RaggedGather/ReadVariableOp:value:0	token_ids#RaggedGather/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
RaggedGather/GatherV2?
IdentityIdentityRaggedGather/GatherV2:output:0^RaggedGather/ReadVariableOp*
T0*#
_output_shapes
:?????????2

Identity}

Identity_1Identitytoken_ids_1^RaggedGather/ReadVariableOp*
T0	*#
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????:?????????: 2:
RaggedGather/ReadVariableOpRaggedGather/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	token_ids:NJ
#
_output_shapes
:?????????
#
_user_specified_name	token_ids
?#
?
qRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4686?
?raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
?
?raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_sub	u
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
??mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2v
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0?
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2v
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1?
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = 2v
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2?
mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_sub*
T
2	*
_output_shapes
 2o
mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2q
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityxRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0n^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1zRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????2?
mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertmRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?
?
@RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4723g
craggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_all
d
`raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice_1	b
^raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice	D
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1
??<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor2E
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0?
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2E
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1?
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (RaggedFromRowSplits/strided_slice_1:0) = 2E
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2?
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (RaggedFromRowSplits/strided_slice:0) = 2E
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4?
<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertcraggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_allLRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0`raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice_1LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0^raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice*
T

2		*
_output_shapes
 2>
<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentitycraggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_all=^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2@
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityGRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0=^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1IRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2|
<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
ARaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_4835m
iraggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_assert_equal_1_all
G
Craggedfromrowsplits_1_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_1_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1
?
<RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2>
<RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOp?
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_assert_equal_1_all=^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2D
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1"?
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
__inference_detokenize_4214
	tokenized	0
,none_export_lookuptableexportv2_table_handle
identity??Assert/Assert?None_Export/LookupTableExportV2?
None_Export/LookupTableExportV2LookupTableExportV2,none_export_lookuptableexportv2_table_handle*
Tkeys0*
Tvalues0	*
_output_shapes

::2!
None_Export/LookupTableExportV2?
EnsureShapeEnsureShape&None_Export/LookupTableExportV2:keys:0*
T0*#
_output_shapes
:?????????*
shape:?????????2
EnsureShape?
EnsureShape_1EnsureShape(None_Export/LookupTableExportV2:values:0*
T0	*#
_output_shapes
:?????????*
shape:?????????2
EnsureShape_1g
argsort/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
argsort/axisg
argsort/NegNegEnsureShape_1:output:0*
T0	*#
_output_shapes
:?????????2
argsort/Neg]
argsort/ShapeShapeargsort/Neg:y:0*
T0	*
_output_shapes
:2
argsort/Shape?
argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
argsort/strided_slice/stack?
argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
argsort/strided_slice/stack_1?
argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
argsort/strided_slice/stack_2?
argsort/strided_sliceStridedSliceargsort/Shape:output:0$argsort/strided_slice/stack:output:0&argsort/strided_slice/stack_1:output:0&argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
argsort/strided_slice^
argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2
argsort/Rank?
argsort/TopKV2TopKV2argsort/Neg:y:0argsort/strided_slice:output:0*
T0	*2
_output_shapes 
:?????????:?????????2
argsort/TopKV2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2EnsureShape_1:output:0argsort/TopKV2:indices:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:?????????2

GatherV2d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2EnsureShape:output:0argsort/TopKV2:indices:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:?????????2

GatherV2_1t
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceGatherV2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yb
EqualEqualstrided_slice:output:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equalx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceGatherV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceGatherV2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2
strided_slice_2s
subSubstrided_slice_1:output:0strided_slice_2:output:0*
T0	*#
_output_shapes
:?????????2
subX
	Equal_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
	Equal_1/yf
Equal_1Equalsub:z:0Equal_1/y:output:0*
T0	*#
_output_shapes
:?????????2	
Equal_1X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstF
AllAllEqual_1:z:0Const:output:0*
_output_shapes
: 2
AllI
and
LogicalAnd	Equal:z:0All:output:0*
_output_shapes
: 2
and?
Assert/ConstConst*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`2
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`2
Assert/Assert/data_0r
Assert/AssertAssertand:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/AssertZ
SizeSizeGatherV2_1:output:0^Assert/Assert*
T0*
_output_shapes
: 2
SizeS
CastCastSize:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Castm
MinimumMinimum	tokenizedCast:y:0*
T0	*0
_output_shapes
:??????????????????2	
Minimump
concat/values_1Const*
_output_shapes
:*
dtype0*
valueBB[UNK]2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2GatherV2_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:?????????2
concatd
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2concat:output:0Minimum:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:??????????????????2

GatherV2_2?
RaggedFromTensor/ShapeShapeGatherV2_2:output:0*
T0*
_output_shapes
:*
out_type0	2
RaggedFromTensor/Shape?
$RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$RaggedFromTensor/strided_slice/stack?
&RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&RaggedFromTensor/strided_slice/stack_1?
&RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&RaggedFromTensor/strided_slice/stack_2?
RaggedFromTensor/strided_sliceStridedSliceRaggedFromTensor/Shape:output:0-RaggedFromTensor/strided_slice/stack:output:0/RaggedFromTensor/strided_slice/stack_1:output:0/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2 
RaggedFromTensor/strided_slice?
&RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&RaggedFromTensor/strided_slice_1/stack?
(RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_1/stack_1?
(RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_1/stack_2?
 RaggedFromTensor/strided_slice_1StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_1/stack:output:01RaggedFromTensor/strided_slice_1/stack_1:output:01RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2"
 RaggedFromTensor/strided_slice_1?
&RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&RaggedFromTensor/strided_slice_2/stack?
(RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_2/stack_1?
(RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_2/stack_2?
 RaggedFromTensor/strided_slice_2StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_2/stack:output:01RaggedFromTensor/strided_slice_2/stack_1:output:01RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2"
 RaggedFromTensor/strided_slice_2?
RaggedFromTensor/mulMul)RaggedFromTensor/strided_slice_1:output:0)RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: 2
RaggedFromTensor/mul?
&RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&RaggedFromTensor/strided_slice_3/stack?
(RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(RaggedFromTensor/strided_slice_3/stack_1?
(RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_3/stack_2?
 RaggedFromTensor/strided_slice_3StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_3/stack:output:01RaggedFromTensor/strided_slice_3/stack_1:output:01RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask2"
 RaggedFromTensor/strided_slice_3?
 RaggedFromTensor/concat/values_0PackRaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:2"
 RaggedFromTensor/concat/values_0~
RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
RaggedFromTensor/concat/axis?
RaggedFromTensor/concatConcatV2)RaggedFromTensor/concat/values_0:output:0)RaggedFromTensor/strided_slice_3:output:0%RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:2
RaggedFromTensor/concat?
RaggedFromTensor/ReshapeReshapeGatherV2_2:output:0 RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:?????????2
RaggedFromTensor/Reshape?
&RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&RaggedFromTensor/strided_slice_4/stack?
(RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_4/stack_1?
(RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_4/stack_2?
 RaggedFromTensor/strided_slice_4StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_4/stack:output:01RaggedFromTensor/strided_slice_4/stack_1:output:01RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2"
 RaggedFromTensor/strided_slice_4?
&RaggedFromTensor/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&RaggedFromTensor/strided_slice_5/stack?
(RaggedFromTensor/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_5/stack_1?
(RaggedFromTensor/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(RaggedFromTensor/strided_slice_5/stack_2?
 RaggedFromTensor/strided_slice_5StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_5/stack:output:01RaggedFromTensor/strided_slice_5/stack_1:output:01RaggedFromTensor/strided_slice_5/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2"
 RaggedFromTensor/strided_slice_5?
1RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape!RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	23
1RaggedFromTensor/RaggedFromUniformRowLength/Shape?
?RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack?
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1?
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2?
9RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlice:RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0HRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0JRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0JRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2;
9RaggedFromTensor/RaggedFromUniformRowLength/strided_slice?
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2T
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y?
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2)RaggedFromTensor/strided_slice_4:output:0[RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: 2R
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add?
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2Z
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start?
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2Z
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta?
WRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/CastCastaRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2Y
WRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast?
YRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1CastaRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2[
YRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1?
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRange[RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast:y:0TRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0]RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2T
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range?
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMul[RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0)RaggedFromTensor/strided_slice_5:output:0*
T0	*#
_output_shapes
:?????????2R
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul?
RaggedSegmentJoin/ShapeShapeTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*
T0	*
_output_shapes
:2
RaggedSegmentJoin/Shape?
%RaggedSegmentJoin/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%RaggedSegmentJoin/strided_slice/stack?
'RaggedSegmentJoin/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedSegmentJoin/strided_slice/stack_1?
'RaggedSegmentJoin/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'RaggedSegmentJoin/strided_slice/stack_2?
RaggedSegmentJoin/strided_sliceStridedSlice RaggedSegmentJoin/Shape:output:0.RaggedSegmentJoin/strided_slice/stack:output:00RaggedSegmentJoin/strided_slice/stack_1:output:00RaggedSegmentJoin/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
RaggedSegmentJoin/strided_slicet
RaggedSegmentJoin/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
RaggedSegmentJoin/sub/y?
RaggedSegmentJoin/subSub(RaggedSegmentJoin/strided_slice:output:0 RaggedSegmentJoin/sub/y:output:0*
T0*
_output_shapes
: 2
RaggedSegmentJoin/sub?
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2?
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_sliceStridedSliceTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2:
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2<
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1?
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/subSubARaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice:output:0CRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????20
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub?
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/ShapeShapeTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*
T0	*
_output_shapes
:*
out_type0	22
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2<
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2?
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y?
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1SubCRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2:output:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 22
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta?
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/CastCast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 27
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast?
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1Cast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 29
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1?
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/rangeRange9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast:y:04RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1:z:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????22
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/range?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/CastCast2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast?
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ShapeShape9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:29
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape?
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlice@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackHRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo:RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast:y:0LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2?
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 29
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const?
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaxMaxFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 27
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max?
;RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2=
;RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumDRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0>RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2;
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2O
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2K
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastRRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2>
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2D
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackKRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2B
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/TileTileERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????28
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2T
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0[RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2N
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2W
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdURaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0^RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2E
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2V
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0SRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2G
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2P
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeGRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2J
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereQRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2F
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2H
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2I
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
%RaggedSegmentJoin/UnsortedSegmentJoinUnsortedSegmentJoin!RaggedFromTensor/Reshape:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin/sub:z:0*
Tindices0	*#
_output_shapes
:?????????*
	separator 2'
%RaggedSegmentJoin/UnsortedSegmentJoin?
StaticRegexReplaceStaticRegexReplace.RaggedSegmentJoin/UnsortedSegmentJoin:output:0*#
_output_shapes
:?????????*
pattern \#\#*
rewrite 2
StaticRegexReplace?
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:?????????*
pattern	^ +| +$*
rewrite 2
StaticRegexReplace_1h
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
value	B B 2
StringSplit/Const?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_1:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:2
StringSplit/StringSplitV2?
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
StringSplit/strided_slice/stack?
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!StringSplit/strided_slice/stack_1?
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!StringSplit/strided_slice/stack_2?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
StringSplit/strided_slice?
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!StringSplit/strided_slice_1/stack?
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_1?
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#StringSplit/strided_slice_1/stack_2?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
StringSplit/strided_slice_1?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2D
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2M
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2R
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2M
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2N
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2L
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2P
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:?????????2Q
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum?
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2O
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2F
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat?
StaticRegexFullMatch_1StaticRegexFullMatch"StringSplit/StringSplitV2:values:0*#
_output_shapes
:?????????*&
pattern\[PAD\]|\[START\]|\[END\]2
StaticRegexFullMatch_1p
LogicalNot_1
LogicalNotStaticRegexFullMatch_1:output:0*#
_output_shapes
:?????????2
LogicalNot_1Z
RaggedMask/assert_equal/NoOpNoOp*
_output_shapes
 2
RaggedMask/assert_equal/NoOp?
RaggedMask/CastCastLogicalNot_1:y:0^RaggedMask/assert_equal/NoOp*

DstT0	*

SrcT0
*#
_output_shapes
:?????????2
RaggedMask/Cast?
 RaggedMask/RaggedReduceSum/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:2"
 RaggedMask/RaggedReduceSum/Shape?
.RaggedMask/RaggedReduceSum/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 20
.RaggedMask/RaggedReduceSum/strided_slice/stack?
0RaggedMask/RaggedReduceSum/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:22
0RaggedMask/RaggedReduceSum/strided_slice/stack_1?
0RaggedMask/RaggedReduceSum/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:22
0RaggedMask/RaggedReduceSum/strided_slice/stack_2?
(RaggedMask/RaggedReduceSum/strided_sliceStridedSlice)RaggedMask/RaggedReduceSum/Shape:output:07RaggedMask/RaggedReduceSum/strided_slice/stack:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_1:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(RaggedMask/RaggedReduceSum/strided_slice?
 RaggedMask/RaggedReduceSum/sub/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2"
 RaggedMask/RaggedReduceSum/sub/y?
RaggedMask/RaggedReduceSum/subSub1RaggedMask/RaggedReduceSum/strided_slice:output:0)RaggedMask/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: 2 
RaggedMask/RaggedReduceSum/sub?
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2I
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2?
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2C
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2E
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1?
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubJRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????29
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub?
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:*
out_type0	2;
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2E
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2?
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0	*
value	B	 R2=
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y?
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubLRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 2;
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta?
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2@
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast?
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2B
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1?
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0=RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????2;
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCast;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast?
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2B
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape?
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2P
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2R
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2R
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceIRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackQRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToCRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2H
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2B
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const?
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 2@
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max?
DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2F
DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumMRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2D
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
?????????2X
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2T
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2G
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :2M
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackTRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2K
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileNRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2A
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2]
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0dRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2W
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2`
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0gRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2N
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2_
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2a
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2U
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0\RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2P
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????2Y
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapePRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2S
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereZRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2O
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2Q
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2W
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2R
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
-RaggedMask/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumRaggedMask/Cast:y:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0"RaggedMask/RaggedReduceSum/sub:z:0*
T0	*
Tindices0	*#
_output_shapes
:?????????2/
-RaggedMask/RaggedReduceSum/UnsortedSegmentSum?
RaggedMask/Cumsum/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 2
RaggedMask/Cumsum/axis?
RaggedMask/CumsumCumsum6RaggedMask/RaggedReduceSum/UnsortedSegmentSum:output:0RaggedMask/Cumsum/axis:output:0*
T0	*#
_output_shapes
:?????????2
RaggedMask/Cumsum?
RaggedMask/concat/values_0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0	*
valueB	R 2
RaggedMask/concat/values_0?
RaggedMask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
?????????2
RaggedMask/concat/axis?
RaggedMask/concatConcatV2#RaggedMask/concat/values_0:output:0RaggedMask/Cumsum:out:0RaggedMask/concat/axis:output:0*
N*
T0	*#
_output_shapes
:?????????2
RaggedMask/concat?
(RaggedMask/RaggedMask/boolean_mask/ShapeShape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
:2*
(RaggedMask/RaggedMask/boolean_mask/Shape?
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 28
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stack?
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1?
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2?
0RaggedMask/RaggedMask/boolean_mask/strided_sliceStridedSlice1RaggedMask/RaggedMask/boolean_mask/Shape:output:0?RaggedMask/RaggedMask/boolean_mask/strided_slice/stack:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:22
0RaggedMask/RaggedMask/boolean_mask/strided_slice?
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2;
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices?
'RaggedMask/RaggedMask/boolean_mask/ProdProd9RaggedMask/RaggedMask/boolean_mask/strided_slice:output:0BRaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2)
'RaggedMask/RaggedMask/boolean_mask/Prod?
*RaggedMask/RaggedMask/boolean_mask/Shape_1Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
:2,
*RaggedMask/RaggedMask/boolean_mask/Shape_1?
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2?
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask24
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1?
*RaggedMask/RaggedMask/boolean_mask/Shape_2Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
:2,
*RaggedMask/RaggedMask/boolean_mask/Shape_2?
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2:
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: 2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1?
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:2<
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2?
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_2:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask24
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2?
2RaggedMask/RaggedMask/boolean_mask/concat/values_1Pack0RaggedMask/RaggedMask/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:24
2RaggedMask/RaggedMask/boolean_mask/concat/values_1?
.RaggedMask/RaggedMask/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 20
.RaggedMask/RaggedMask/boolean_mask/concat/axis?
)RaggedMask/RaggedMask/boolean_mask/concatConcatV2;RaggedMask/RaggedMask/boolean_mask/strided_slice_1:output:0;RaggedMask/RaggedMask/boolean_mask/concat/values_1:output:0;RaggedMask/RaggedMask/boolean_mask/strided_slice_2:output:07RaggedMask/RaggedMask/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)RaggedMask/RaggedMask/boolean_mask/concat?
*RaggedMask/RaggedMask/boolean_mask/ReshapeReshape"StringSplit/StringSplitV2:values:02RaggedMask/RaggedMask/boolean_mask/concat:output:0*
T0*#
_output_shapes
:?????????2,
*RaggedMask/RaggedMask/boolean_mask/Reshape?
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
?????????24
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape?
,RaggedMask/RaggedMask/boolean_mask/Reshape_1ReshapeLogicalNot_1:y:0;RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2.
,RaggedMask/RaggedMask/boolean_mask/Reshape_1?
(RaggedMask/RaggedMask/boolean_mask/WhereWhere5RaggedMask/RaggedMask/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2*
(RaggedMask/RaggedMask/boolean_mask/Where?
*RaggedMask/RaggedMask/boolean_mask/SqueezeSqueeze0RaggedMask/RaggedMask/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2,
*RaggedMask/RaggedMask/boolean_mask/Squeeze?
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : 22
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axis?
+RaggedMask/RaggedMask/boolean_mask/GatherV2GatherV23RaggedMask/RaggedMask/boolean_mask/Reshape:output:03RaggedMask/RaggedMask/boolean_mask/Squeeze:output:09RaggedMask/RaggedMask/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2-
+RaggedMask/RaggedMask/boolean_mask/GatherV2?
RaggedSegmentJoin_1/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:2
RaggedSegmentJoin_1/Shape?
'RaggedSegmentJoin_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'RaggedSegmentJoin_1/strided_slice/stack?
)RaggedSegmentJoin_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedSegmentJoin_1/strided_slice/stack_1?
)RaggedSegmentJoin_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)RaggedSegmentJoin_1/strided_slice/stack_2?
!RaggedSegmentJoin_1/strided_sliceStridedSlice"RaggedSegmentJoin_1/Shape:output:00RaggedSegmentJoin_1/strided_slice/stack:output:02RaggedSegmentJoin_1/strided_slice/stack_1:output:02RaggedSegmentJoin_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!RaggedSegmentJoin_1/strided_slicex
RaggedSegmentJoin_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
RaggedSegmentJoin_1/sub/y?
RaggedSegmentJoin_1/subSub*RaggedSegmentJoin_1/strided_slice:output:0"RaggedSegmentJoin_1/sub/y:output:0*
T0*
_output_shapes
: 2
RaggedSegmentJoin_1/sub?
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2?
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceRaggedMask/concat:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*
end_mask2<
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2?
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceRaggedMask/concat:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask2>
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1?
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/subSubCRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice:output:0ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:?????????22
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub?
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:*
out_type0	24
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2?
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2>
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2?
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R26
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y?
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1SubERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: 24
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta?
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/CastCastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: 29
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast?
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1CastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2;
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1?
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/rangeRange;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast:y:06RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1:z:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:?????????24
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/CastCast4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast?
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ShapeShape;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
:2;
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape?
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack?
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1?
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2?
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceBRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackJRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape?
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:?????????2A
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo?
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const?
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: 29
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max?
=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2?
=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x?
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumFRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum/x:output:0@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: 2=
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:?????????2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range?
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2Q
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim?
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2M
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastTRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:??????????????????2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim?
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????2@
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims?
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :2F
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0?
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackMRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:2D
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples?
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/TileTileGRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:??????????????????2:
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape?
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2V
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2?
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0]RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2P
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice?
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2Y
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices?
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdWRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0`RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2G
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
:2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2?
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2X
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1?
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Z
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1?
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis?
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0URaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2I
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:?????????2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape?
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2R
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape?
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeIRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2L
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1?
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereSRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2H
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where?
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2J
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze?
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis?
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2K
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2?
'RaggedSegmentJoin_1/UnsortedSegmentJoinUnsortedSegmentJoin4RaggedMask/RaggedMask/boolean_mask/GatherV2:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin_1/sub:z:0*
Tindices0	*#
_output_shapes
:?????????*
	separator 2)
'RaggedSegmentJoin_1/UnsortedSegmentJoin?
IdentityIdentity0RaggedSegmentJoin_1/UnsortedSegmentJoin:output:0^Assert/Assert ^None_Export/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2
Assert/AssertAssert/Assert2B
None_Export/LookupTableExportV2None_Export/LookupTableExportV2:[ W
0
_output_shapes
:??????????????????
#
_user_specified_name	tokenized
?$
?
sRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5039?
?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
?
?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
??oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2x
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0?
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2x
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1?
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = 2x
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2?
oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_sub*
T
2	*
_output_shapes
 2q
oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0p^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2u
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????2?
oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?$
?
\RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5003?
?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_all
?
?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_strided_slice	?
?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
??XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2a
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0?
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2a
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1?
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = 2a
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2?
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = 2a
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4?
XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_const*
T

2		*
_output_shapes
 2Z
XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0Y^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2^
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
9
__inference__creator_5339
identity??
hash_table?

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*O
shared_name@>hash_table_/content/drive/MyDrive/CSC 582/wiki_vocab.txt_-2_-1*
value_dtype0	2

hash_tablei
IdentityIdentityhash_table:table_handle:0^hash_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
[RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4762?
?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
?
VRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2X
VRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp?
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2^
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
BRaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_4836k
graggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_all
h
draggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice_1	f
braggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice	F
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1
??>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert?
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor2G
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_0?
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2G
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_1?
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_1/strided_slice_1:0) = 2G
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_2?
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_1/strided_slice:0) = 2G
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_4?
>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_allNRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice_1NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice*
T

2		*
_output_shapes
 2@
>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert?
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_all?^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity:output:0?^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2D
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1"?
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
rRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5038?
?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
?
mRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2o
mRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp?
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2u
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?
e
__inference_lookup_4605
	token_ids	
gather_resource:
??

identity_1??Gather?
GatherResourceGathergather_resource	token_ids*
Tindices0	*0
_output_shapes
:??????????????????*
dtype02
Gatherl
IdentityIdentityGather:output:0*
T0*0
_output_shapes
:??????????????????2

Identity{

Identity_1IdentityIdentity:output:0^Gather*
T0*0
_output_shapes
:??????????????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2
GatherGather:[ W
0
_output_shapes
:??????????????????
#
_user_specified_name	token_ids
?

?
8RaggedConcat_assert_equal_3_Assert_AssertGuard_true_5265[
Wraggedconcat_assert_equal_3_assert_assertguard_identity_raggedconcat_assert_equal_3_all
>
:raggedconcat_assert_equal_3_assert_assertguard_placeholder	@
<raggedconcat_assert_equal_3_assert_assertguard_placeholder_1	=
9raggedconcat_assert_equal_3_assert_assertguard_identity_1
?
3RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 25
3RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOp?
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentityWraggedconcat_assert_equal_3_assert_assertguard_identity_raggedconcat_assert_equal_3_all4^RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 29
7RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity?
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2;
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1"
9raggedconcat_assert_equal_3_assert_assertguard_identity_1BRaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?#
?
ZRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4650?
?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_all
?
?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_strided_slice	?
?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_const	^
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
??VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2_
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0?
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2_
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1?
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = 2_
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2?
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = 2_
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4?
VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_strided_slicefRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_const*
T

2		*
_output_shapes
 2X
VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2Z
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0W^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1cRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertVRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
sRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4799?
?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
?
?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
??oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2x
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0?
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2x
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1?
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = 2x
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2?
oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_sub*
T
2	*
_output_shapes
 2q
oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0p^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2u
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????2?
oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?$
?
sRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4912?
?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
?
?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
??oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*X
valueOBM BGArguments to from_row_splits do not form a valid RaggedTensor:monotonic2x
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0?
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:2x
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1?
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = 2x
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2?
oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_sub*
T
2	*
_output_shapes
 2q
oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert?
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2s
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity?
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0p^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2u
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1"?
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :?????????2?
oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :)%
#
_output_shapes
:?????????
?
?
__inference__initializer_5346*
&text_file_id_table_init_asset_filepathF
Btext_file_id_table_init_initializetablefromtextfilev2_table_handle
identity??5text_file_id_table_init/InitializeTableFromTextFileV2?
5text_file_id_table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2Btext_file_id_table_init_initializetablefromtextfilev2_table_handle&text_file_id_table_init_asset_filepath*
_output_shapes
 *
	key_index?????????*
value_index?????????27
5text_file_id_table_init/InitializeTableFromTextFileV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:06^text_file_id_table_init/InitializeTableFromTextFileV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2n
5text_file_id_table_init/InitializeTableFromTextFileV25text_file_id_table_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
?
?
YRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4649?
?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_all
_
[raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	a
]raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	^
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
?
TRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2V
TRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp?
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allU^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2Z
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1cRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
?RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4722i
eraggedfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_assert_equal_1_all
E
Araggedfromrowsplits_assert_equal_1_assert_assertguard_placeholder	G
Craggedfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	D
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1
?
:RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2<
:RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp?
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityeraggedfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_assert_equal_1_all;^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2@
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityGRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1IRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_5358*
&text_file_id_table_init_asset_filepathF
Btext_file_id_table_init_initializetablefromtextfilev2_table_handle
identity??5text_file_id_table_init/InitializeTableFromTextFileV2?
5text_file_id_table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2Btext_file_id_table_init_initializetablefromtextfilev2_table_handle&text_file_id_table_init_asset_filepath*
_output_shapes
 *
	key_index?????????*
value_index?????????27
5text_file_id_table_init/InitializeTableFromTextFileV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
Const?
IdentityIdentityConst:output:06^text_file_id_table_init/InitializeTableFromTextFileV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2n
5text_file_id_table_init/InitializeTableFromTextFileV25text_file_id_table_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
?
a
__inference_get_vocab_size_4598-
shape_readvariableop_resource:
??
identity??
Shape/ReadVariableOpReadVariableOpshape_readvariableop_resource*
_output_shapes

:??*
dtype02
Shape/ReadVariableOpZ
ShapeConst*
_output_shapes
:*
dtype0*
valueB:??2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceY
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 
?

?
8RaggedConcat_assert_equal_1_Assert_AssertGuard_true_5235[
Wraggedconcat_assert_equal_1_assert_assertguard_identity_raggedconcat_assert_equal_1_all
>
:raggedconcat_assert_equal_1_assert_assertguard_placeholder	@
<raggedconcat_assert_equal_1_assert_assertguard_placeholder_1	=
9raggedconcat_assert_equal_1_assert_assertguard_identity_1
?
3RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 25
3RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOp?
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentityWraggedconcat_assert_equal_1_assert_assertguard_identity_raggedconcat_assert_equal_1_all4^RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 29
7RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity?
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2;
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1"
9raggedconcat_assert_equal_1_assert_assertguard_identity_1BRaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
\RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4763?
?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_all
?
?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_strided_slice	?
?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
??XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2a
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0?
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2a
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1?
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = 2a
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2?
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = 2a
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4?
XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_const*
T

2		*
_output_shapes
 2Z
XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0Y^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2^
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
ARaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_true_4948m
iraggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_assert_equal_1_all
G
Craggedfromrowsplits_2_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_2_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1
?
<RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2>
<RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOp?
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_assert_equal_1_all=^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2B
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity?
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2D
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1"?
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
\RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4876?
?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_all
?
?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_strided_slice	?
?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
??XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero2a
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0?
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:2a
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1?
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = 2a
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2?
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = 2a
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4?
XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssert?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_const*
T

2		*
_output_shapes
 2Z
XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert?
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0Y^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2^
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
[RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4875?
?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
?
VRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2X
VRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp?
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity?raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2\
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity?
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2^
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1"?
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp2,

asset_path_initializer:0wiki_vocab.txt2.

asset_path_initializer_1:0wiki_vocab.txt:?
6
en

signatures"
_generic_user_object
?
	tokenizer
_reserved_tokens
_vocab_path
	vocab

detokenize
get_reserved_tokens
get_vocab_path
get_vocab_size

lookup
tokenize"
_generic_user_object
"
signature_map
N
_basic_tokenizer
_wordpiece_tokenizer"
_generic_user_object
 "
trackable_list_wrapper
* 
:??2Variable
"
_generic_user_object
7
	_vocab_lookup_table"
_generic_user_object
R

_initializer
_create_resource
_initialize
_destroy_resourceR 
-
	_filename"
_generic_user_object
*
?2?
__inference_detokenize_4214
__inference_detokenize_4579?
???
FullArgSpec 
args?
jself
j	tokenized
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_get_reserved_tokens_4583?
???
FullArgSpec
args?
jself
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_get_vocab_path_4588?
???
FullArgSpec
args?
jself
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_get_vocab_size_4598?
???
FullArgSpec
args?
jself
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_lookup_4605
__inference_lookup_4615?
???
FullArgSpec 
args?
jself
j	token_ids
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_tokenize_5334?
???
FullArgSpec
args?
jself
	jstrings
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_5339?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_5346?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_5351?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const
J	
Const_1
J	
Const_25
__inference__creator_5339?

? 
? "? 7
__inference__destroyer_5351?

? 
? "? =
__inference__initializer_5346	?

? 
? "? u
__inference_detokenize_4214V	;?8
1?.
,?)
	tokenized??????????????????	
? "???????????
__inference_detokenize_4579s	X?U
N?K
I?F0?-
???????????????????
?	
`
?	RaggedTensorSpec
? "??????????D
$__inference_get_reserved_tokens_4583?

? 
? "?>
__inference_get_vocab_path_4588?

? 
? "? >
__inference_get_vocab_size_4598?

? 
? "? ~
__inference_lookup_4605c;?8
1?.
,?)
	token_ids??????????????????	
? "!????????????????????
__inference_lookup_4615?X?U
N?K
I?F0?-
???????????????????
?	
`
?	RaggedTensorSpec
? "I?F0?-
???????????????????
?
`
?	RaggedTensorSpec?
__inference_tokenize_5334	,?)
"?
?
strings?????????
? "I?F0?-
???????????????????
?	
`
?	RaggedTensorSpec