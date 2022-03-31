import os
import io
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from azure.keyvault.secrets import SecretClient


####################################################
# Configuration
BLOB_account = 'happywhale2'
BLOB_container = 'happywhale'
BLOB_name = 'out.txt'
FS_fname = 'in.txt'
KV_account = 'happywhale0451856571'
KV_secret_name = 'testsecret'
# Print datetime and environment variables
print(f'{datetime.now()}')
print(f'This is an environment variable: {os.environ.get("public1")}')
print(f'This is a secret environment variable: {os.environ.get("private1")}')
# Authenticate with Azure
# (1) environment variables, (2) Managed Identity, (3) User logged in in Microsoft application, ...
AZ_credential = DefaultAzureCredential()
# Retrieve primary key for blob from the Azure Keyvault
KV_url = f'https://{KV_account}.vault.azure.net'
KV_secretClient = SecretClient(vault_url=KV_url, credential=AZ_credential)
BLOB_PrimaryKey = KV_secretClient.get_secret(KV_secret_name).value
# Set the BLOB client
BLOB_CONN_STR = f'DefaultEndpointsProtocol=https;AccountName={BLOB_account};AccountKey={BLOB_PrimaryKey};EndpointSuffix=core.windows.net'
BLOB_client = BlobClient.from_connection_string(conn_str=BLOB_CONN_STR, container_name=BLOB_container, blob_name=BLOB_name)
# Read text-file from mounted fileshare and write to BLOB
with open(f'mnt/{FS_fname}', 'rb') as f:
    dataBytesBuffer = io.BytesIO(f.read())
dataBytesBuffer.seek(0)
BLOB_client.upload_blob(dataBytesBuffer, overwrite=True)
print(f'File successfully uploaded to blob')


