from pyspark.sql.functions import *
from pyspark.sql.functions import to_json,from_json
from pyspark.sql import SQLContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as sf
from pyspark.sql.types import FloatType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DateType
from pyspark.sql.types import StringType
from pyspark.sql.functions import when, col
from pyspark.sql.functions import explode
from pyspark.sql.functions import countDistinct
from pyspark.ml.feature import Bucketizer
import datetime, time 
from pyspark.sql.functions import unix_timestamp, lit
from pyspark.sql.functions import split,datediff
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.functions import sum,avg,max,min,mean,count,countDistinct
from pyspark.sql.functions import concat
from pyspark.sql.functions import year,quarter,month


#Reading Files


#Joining Encounters Name with Activities
ref_encounters=ref_encounters.select('code','name').drop_duplicates().withColumnRenamed('code','encounterType').withColumnRenamed('name','encounterName')
activities=activities.na.fill(value=0.,subset=["encounterType"])
newRow = spark.createDataFrame([('0','Blanks')], ref_encounters.columns)
ref_encounters=ref_encounters.union(newRow)
activities=activities.join(ref_encounters,['encounterType'],'left')

#Joining Sender Category with Activities
ref_providers=ref_providers.withColumn('SenderCategory', when((col('name').contains('NOSE')) | (col('name').contains('THROAT')) | (col('name').contains('HEARING')),'EAR NOSE THROAT')
                         .when((col('name').contains('EYES')) | (col('name').contains('OPTIC')),'OPTICALS')
                         .when((col('name').contains('DENTAL')) | (col('name').contains('TEETH')),'DENTAL')
                         .when((col('name').contains('HEART')) | (col('name').contains('CARDIAC')) ,'HEART')
                         .when((col('name').contains('NEURO')) | (col('name').contains('PSYCHIATRIC')),'NEURO')
                         .when((col('name').contains('PEDIATRIC')) | (col('name').contains('MATERN')),'MATERNITY/PEDIA')
                         .when((col('name').contains('DIABETES')),'DIABETES')
                         .when((col('name').contains('CANCER')),'CANCER')
                         .when((col('name').contains('SKIN')) | (col('name').contains('DERMA')) | (col('name').contains('HAIR')) ,'DERMA')
                         .when((col('name').contains('KIDS')) | (col('name').contains('CHILD')) ,'KIDS')
                         .when((col('name').contains('PHARMACY')),'PHARMACY')
                         .when((col('name').contains('DAY CARE')) | (col('name').contains('DAYCARE')) |(col('name').contains('OUTPATIENT')),'OUTPATIENT/DAYCARE')
                         .when((col('name').contains('LABORATORY')),'LABORATORY')
                         .when((col('name').contains('DIAGNOSTIC')),'DIAGNOSTIC')
                         .when((col('name').contains('MEDICAL CENTRE')) | (col('name').contains('MEDICAL CENTER')),'MEDICAL CENTRE')
                         .when((col('name').contains('HOSPITAL')) | (col('name').contains('HOSPITEL')),'GENERAL HOSPITAL')
                         .when((col('name').contains('CLINIC')),'CLINIC')
                         .otherwise('OTHERS'))

ref_providers=ref_providers.withColumnRenamed('providerCode','senderId')
activities=activities.join(ref_providers,['senderId'],how='left')

#Keeping only Primary Diagnosis and ICD-1 Merging
activities=activities.withColumn('diagnoses_flat',sf.explode_outer('diagnoses')).drop('diagnoses')
activities=activities.withColumn('Primary_ICD',col("diagnoses_flat.code"))\
.withColumn('ICD_Type',col('diagnoses_flat.type'))
activities=activities[activities['ICD_Type'].isin(['Principal']) | activities['ICD_Type'].isNull()]
icd_level1=icd_level1.drop('_c0')
activities=activities.withColumn("ICD1", split(col("Primary_ICD"), "\.").getItem(0)).drop('Primary_ICD')
activities=activities.join(icd_level1,['ICD1'],how='left')

#Adding Clinician Information to Activities
activities=activities.withColumn('ClinicianID',col('activities.clinician.id'))\
.withColumn('ClinicianName',col('activities.clinician.name'))
activities=activities.join(clinicianInfo,['ClinicianID'],how='left')

#Defining Remittance Date Column
activities=activities.withColumn('RemittanceDate',when(col('resubmissionCount')==0,col('firstRemittanceDate'))
                     .otherwise(col('lastRemittanceDate')))


#Adding Activity Code, Activity Type , Act Code Name and FirstSubmission(Act)
activities=activities.withColumn('act_code',activities.activities.code)\
.withColumn('act_type',activities.activities.type)\
.withColumn('act_codeName',activities.activities.codeName)\
.withColumn('act_firstSubmissionDate',activities.activities.firstSubmissionDate)\
.withColumn('act_claimed_amt',activities.activities.net)\
.withColumn('act_denied_amt',activities.activities.deniedAmount)\
.withColumn('act_denial_code',activities.activities.latestDenialCode)\
.withColumn('act_denial_name',activities.activities.latestDenialCodeName)\
.withColumn('act_payment_amt',activities.activities.paymentAmount)\
.withColumn('act_ClinicianID',col('activities.clinician.id'))\
.withColumn('act_ClinicianName',col('activities.clinician.name'))



#Defining Category, Detailed Category as per act_type and act_code, defining versionCode and authCode

activities=activities.withColumn('versionCode',when((col('authorityCode')=='DHA') & (col('firstSubmissionDate') <=(lit('2020-09-01 00:00:00'))) ,"v2015")
                                   .when((col('authorityCode')=='DHA') & (col('firstSubmissionDate') >(lit('2020-09-01 00:00:00'))) ,"cpt4_v2018")
                                   .when((col('authorityCode')=='HAAD') & (col('firstSubmissionDate') <=(lit('2021-07-01 00:00:00'))) ,"v2015")
                                       .otherwise('cpt4_v2018'))

activities=activities.withColumn('act_type',activities['act_type'].cast(StringType()))
ref_act=ref_act.withColumnRenamed('activityType','act_type').withColumnRenamed('activityCode','act_code').withColumnRenamed('shortDescription','act_shortDescription')
 
activities=activities.join(ref_act,on=['act_code','act_type','authorityCode','versionCode'],how='left')
activities=activities.withColumn("act_type_name",when(col('act_type')=='3','CPT_CODE').
                                      when(col('act_type')=='4','HCPCS_CODE').
                                      when(col('act_type')=='5','DRUGS_CODE').
                                      when(col('act_type')=='6','HCPCS_CODE').
                                      when(col('act_type')=='8','SERVICE_CODE').                                      
                                      when(col('act_type')=='9','CONSULTATION').
                                      when(col('act_type')=='10','CONSULTATION').
                                      otherwise('OTHER_CODES'))

CPT_Master_detailed=CPT_Master_detailed.withColumn('act_type',lit('3')).withColumnRenamed('CPTs','act_code')
activities=activities.join(CPT_Master_detailed,on=['act_code','act_type'],how='left')

#Category Detailed Category
activities=activities.withColumn("Category",when(col('act_type')=='3',col('Category')).
                                      when(col('act_type')=='4','HCPCS_CODE').
                                      when(col('act_type')=='5','DRUGS_CODE').
                                      when(col('act_type')=='6','HCPCS_CODE').
                                      when(col('act_type')=='8','SERVICE_CODE').                                      
                                      when(col('act_type')=='9','CONSULTATION').
                                      when(col('act_type')=='10','CONSULTATION').
                                      otherwise('OTHER_CODES'))
activities=activities.withColumn("Detailed_Category",when(col('act_type')=='3',col('Detailed_Category')).otherwise(col('act_shortDescription')))
activities=activities.withColumn('Category',when((col('act_type')=='3') & (col('Category').isNull()),'OTHER_CPT_CODES').otherwise(col('Category')))
activities=activities.withColumn("Detailed_Category",when((col('act_type')=='3') & (col('Detailed_Category').isNull()),'OTHER_CPTs').
                                 when((col('act_type')=='4') & (col('Detailed_Category').isNull()),'OTHER_HCPCS_C').
                                 when((col('act_type')=='5') & (col('Detailed_Category').isNull()),'OTHER_DRUG_C').
                                 when((col('act_type')=='6') & (col('Detailed_Category').isNull()),'OTHER_HCPCS_C').
                                 when((col('act_type')=='8') & (col('Detailed_Category').isNull()),'SERVICE_CODE').
                                 when((col('act_type')=='9') & (col('Detailed_Category').isNull()),'CONSULTATION'). 
                                 when((col('act_type')=='10') & (col('Detailed_Category').isNull()),'CONSULTATION').
                                 otherwise('BLANKS'))

#Defining year, month, quarter
activities=activities.withColumn('encounterEndYear',year(activities.encounterEndDate))
activities=activities.withColumn('encounterEndQuarter',quarter(activities.encounterEndDate))
activities=activities.withColumn('encounterEndMonth',month(activities.encounterEndDate))

#Defining First Payment_Lead_Time and First SubmissionLeadTime
activities=activities.withColumn('paymentLeadTime',datediff(activities['firstRemittanceDate'],activities['act_firstSubmissionDate']))
activities=activities.withColumn('submissionLeadTime',datediff(activities['act_firstSubmissionDate'],activities['encounterEndDate']))


#Denial Type
activities=activities.na.fill(value="EMPTY",subset=["act_denial_code"])
activities=activities.withColumn('DenialType', when(col('act_denial_code')=="EMPTY","None")
                         .when(col('act_denial_code').contains('MNEC'),'Medical Denial').otherwise('Technical Denial'))

#unique_claimids
activities=activities.withColumn('claimId',concat(col("claimId"),sf.lit("--"), col("senderId")))



activities=activities.withColumn('submissionDateExists',when(col('firstSubmissionDate').isNull(),sf.lit("EMPTY")).otherwise(sf.lit('EXISTS')))
activities=activities.withColumn("ErrorRecord",when(col('act_denied_amt')<0,sf.lit("Neg_Denied")).\
                                 when(col('act_payment_amt')<0,sf.lit("Neg_Payment")).\
                                 when(col('act_claimed_amt')<=0,sf.lit("Neg_Claimed")).\
                                 when(col('act_claimed_amt')<col('act_payment_amt'),sf.lit("payment_>_claimed")).otherwise(sf.lit('NO')))

activities=activities.na.fill(value=0.,subset=["act_denied_amt","act_payment_amt","act_claimed_amt"])
activities=activities.na.fill(value="EMPTY",subset=["receiverName","encounterName","Category","act_denial_code","encounterEndYear"])

#FirstRemittanceData
activities=activities.withColumn('remittanceAdvices',col('activities.remittanceAdvices'))
activities=activities.withColumn('remittanceAdvices_flat',sf.explode_outer('remittanceAdvices')).drop('remittanceAdvices')
 
activities=activities.drop_duplicates(subset=['claimId','activities'])
activities=activities.withColumn('FirstdenialCode',activities.remittanceAdvices_flat.denialCode)\
.withColumn('firstdenialCodeName',activities.remittanceAdvices_flat.denialCodeName)\
.withColumn('firstPaymentAmount',activities.remittanceAdvices_flat.paymentAmount)

dashboard_file=activities.groupby('claimId','act_code',
 'act_type','ICD1',
 'ClinicianID',
 'senderId',
 'encounterType','encounterStartType','hasPendingSubmission',
 'hasSubmissionError',
 'lastRemittanceDate',
 'lastSubmissionDate',
 'payerId',
 'payerName',
 'paymentLeadTime',
 'receiverId',
 'receiverName',
 'resubmissionCount',
 'encounterName',
 'act_firstSubmissionDate',
 'act_claimed_amt',
 'act_denied_amt',
 'act_denial_code',
 'act_denial_name',
 'act_payment_amt',
 'act_ClinicianID',
 'act_ClinicianName',
 'name',
 'SenderCategory',
 'RemittanceDate',
 'ICD_Type',
 'ClinicianName',
 'ClinicianProfession',
 'ClinicianMajor',
 'ClinicianCategory',
 'ICD_1_Description',
 'act_shortDescription',
 'act_type_name',
 'Category',
 'Detailed_Category',
 'encounterEndYear',
 'encounterEndQuarter',
 'encounterEndMonth',
 'submissionLeadTime',
 'ErrorRecord',
 'submissionDateExists','firstDenialType','firstPaymentAmount','FirstdenialCode','firstdenialCodeName','FirstRem_ErrorRecord','firstDeniedAmount',
'DenialType').agg(sum('act_claimed_amt').alias('Claimed'),sum('act_payment_amt').alias('Paid'),sum('act_denied_amt').alias('Denied'),avg('paymentLeadTime').alias('avgPaymentTime'),avg('submissionLeadTime').alias('avgSubmTime'))
