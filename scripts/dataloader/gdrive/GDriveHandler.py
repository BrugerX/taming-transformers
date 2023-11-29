import os.path

import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io
from PIL import Image
import numpy as np



class GDrive_Handler:
    """
    @arg scopes: The scopes of the API service
    @arg credentials_path: Client credentials.
    @arg write_new_token: If set to True it will write a new access token based off of the scopes. This will open a pop-up window requesting authorization of your email.

    If set to false, it will use the old access token, which were created using the previous scopes.

    """

    def __init__(self, scopes, credentials_path, write_new_token=True, token_path="token.json"):
        self.credentials = self.get_drive_credentials(credentials_path,scopes,write_new_token=write_new_token,token_path=token_path)
        self.service = build("drive", "v3", credentials=self.credentials)


    """
    Gets the actual google drive credentials based off of the client credentials.
    
    Can also save an optional access token.json file, if write_new_token = false. However this does not work for all types of scopes.
    
    """
    def get_drive_credentials(self,credentials_path,SCOPES,write_new_token = True,token_path = "json.org"):
      """Shows basic usage of the Drive v3 API.
      Prints the names and ids of the first 10 files the user has access to.
      """
      creds = None
      # The file token.json stores the user's access and refresh tokens, and is
      # created automatically when the authorization flow completes for the first
      # time.
      if os.path.exists(token_path) and not write_new_token:
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
      # If there are no (valid) credentials available, let the user log in.
      if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
          creds.refresh(Request())
        else:
          flow = InstalledAppFlow.from_client_secrets_file(
              credentials_path, SCOPES
          )
          creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, "w") as token:
          token.write(creds.to_json())

        return creds

    """Returns an image in array form from Google Drive
    
    @Precondition: real_file_id is an ID of a Google Drive file with an extension supported by PILLOW
    
    """
    def download_image_file(self, real_file_id):
        try:
            # Create drive api client
            file_id = real_file_id

            request = self.service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")

            # Convert the downloaded bytes to an image
            file.seek(0)  # Go to the beginning of the IO object
            image = Image.open(file)

            # Convert the image to a numpy array
            image_array = np.array(image)

            return image_array

        except HttpError as error:
            # Print more detailed error information
            print(f"An error occurred: {error}")
            if error.resp.status in [403, 500, 503]:
                print(f"Reason: {error.resp.reason}")
                print(f"Body: {error.resp.body}")
                # Try to parse and print out detailed error message
                try:
                    error_details = json.loads(error.resp.body.decode("utf-8"))
                    print(json.dumps(error_details, indent=2))
                except json.JSONDecodeError:
                    print("Could not parse error details.")
            return None