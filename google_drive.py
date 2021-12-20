from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build


def auth():
    from google.colab import auth
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)


def save_model(model_name):
    drive_service = build('drive', 'v3')
    file_metadata = {'name': model_name}
    media = MediaFileUpload(model_name, resumable=True)
    created = drive_service.files().create(body=file_metadata,
                                           media_body=media,
                                           fields='id').execute()
    return created.get('id')
