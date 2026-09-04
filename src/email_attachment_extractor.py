import os
import imaplib
import email
import hashlib
from email.header import decode_header
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


EMAIL_ADDRESS = os.getenv(
    "EMAIL_ADDRESS"
)

EMAIL_APP_PASSWORD = os.getenv(
    "EMAIL_APP_PASSWORD"
)

IMAP_SERVER = os.getenv(
    "IMAP_SERVER",
    "imap.gmail.com"
)


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# ATTACHMENT FOLDER
# ============================================================

ATTACHMENT_FOLDER = os.path.join(
    BASE_DIR,
    "attachments"
)


os.makedirs(
    ATTACHMENT_FOLDER,
    exist_ok=True
)


# ============================================================
# SUPPORTED ATTACHMENTS
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".docx",
    ".txt",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================================
# DECODE EMAIL SUBJECT
# ============================================================

def decode_subject(subject):

    if not subject:

        return "No Subject"

    decoded_parts = decode_header(
        subject
    )

    result = ""

    for part, encoding in decoded_parts:

        if isinstance(part, bytes):

            try:

                result += part.decode(
                    encoding or "utf-8",
                    errors="ignore"
                )

            except Exception:

                result += part.decode(
                    "utf-8",
                    errors="ignore"
                )

        else:

            result += str(part)

    return result


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(filename):

    if not filename:

        return None

    filename = os.path.basename(
        filename
    )

    invalid_characters = (
        '<>:"/\\|?*'
    )

    for character in invalid_characters:

        filename = filename.replace(
            character,
            "_"
        )

    return filename.strip()


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(data):

    """
    Creates a unique hash from attachment content.

    This prevents the same file from being
    downloaded multiple times even if the
    filename is different.
    """

    return hashlib.sha256(
        data
    ).hexdigest()


# ============================================================
# CHECK IF FILE ALREADY EXISTS
# ============================================================

def attachment_already_exists(
    filename,
    attachment_data
):

    # --------------------------------------------------------
    # First check exact filename
    # --------------------------------------------------------

    file_path = os.path.join(
        ATTACHMENT_FOLDER,
        filename
    )

    if os.path.exists(
        file_path
    ):

        return True


    # --------------------------------------------------------
    # Check file content
    # --------------------------------------------------------

    new_hash = calculate_file_hash(
        attachment_data
    )


    try:

        for existing_file in os.listdir(
            ATTACHMENT_FOLDER
        ):

            existing_path = os.path.join(
                ATTACHMENT_FOLDER,
                existing_file
            )

            if not os.path.isfile(
                existing_path
            ):

                continue


            try:

                with open(
                    existing_path,
                    "rb"
                ) as file:

                    existing_data = file.read()


                existing_hash = (
                    calculate_file_hash(
                        existing_data
                    )
                )


                if existing_hash == new_hash:

                    return True


            except Exception:

                continue


    except Exception:

        pass


    return False


# ============================================================
# GET UNIQUE FILE PATH
# ============================================================

def get_unique_file_path(
    filename
):

    file_path = os.path.join(
        ATTACHMENT_FOLDER,
        filename
    )


    if not os.path.exists(
        file_path
    ):

        return file_path


    base_name, extension = (
        os.path.splitext(
            filename
        )
    )


    counter = 1


    while True:

        new_filename = (
            f"{base_name}_{counter}{extension}"
        )


        new_path = os.path.join(
            ATTACHMENT_FOLDER,
            new_filename
        )


        if not os.path.exists(
            new_path
        ):

            return new_path


        counter += 1


# ============================================================
# DOWNLOAD ATTACHMENTS
# ============================================================

def download_attachments():

    print("=" * 70)

    print(
        "HR EMAIL ATTACHMENT EXTRACTOR"
    )

    print("=" * 70)


    # ========================================================
    # CHECK EMAIL CONFIGURATION
    # ========================================================

    if not EMAIL_ADDRESS:

        print(
            "\n❌ EMAIL_ADDRESS is missing "
            "from .env"
        )

        return


    if not EMAIL_APP_PASSWORD:

        print(
            "\n❌ EMAIL_APP_PASSWORD is missing "
            "from .env"
        )

        return


    # ========================================================
    # CONNECT TO GMAIL
    # ========================================================

    print(
        "\nConnecting to Gmail..."
    )


    try:

        mail = imaplib.IMAP4_SSL(
            IMAP_SERVER
        )


        mail.login(
            EMAIL_ADDRESS,
            EMAIL_APP_PASSWORD
        )


        print(
            "✅ Gmail login successful"
        )


    except Exception as e:

        print(
            "\n❌ Gmail connection failed:"
        )

        print(e)

        return


    # ========================================================
    # OPEN INBOX
    # ========================================================

    try:

        status, _ = mail.select(
            "INBOX"
        )


        if status != "OK":

            print(
                "❌ Could not open inbox."
            )

            mail.logout()

            return


        print(
            "✅ Inbox opened"
        )


    except Exception as e:

        print(
            "\n❌ Inbox error:"
        )

        print(e)

        mail.logout()

        return


    # ========================================================
    # SEARCH EMAILS
    # ========================================================

    try:

        status, messages = mail.search(
            None,
            "ALL"
        )


    except Exception as e:

        print(
            "\n❌ Email search failed:"
        )

        print(e)

        mail.logout()

        return


    if status != "OK":

        print(
            "\n❌ Could not retrieve emails."
        )

        mail.logout()

        return


    email_ids = messages[0].split()


    print(
        f"\n📧 Emails found: "
        f"{len(email_ids)}"
    )


    # ========================================================
    # TRACK EMAILS PROCESSED IN THIS RUN
    # ========================================================

    processed_message_ids = set()


    # ========================================================
    # STATISTICS
    # ========================================================

    saved_count = 0

    duplicate_count = 0

    unsupported_count = 0

    empty_count = 0


    # ========================================================
    # PROCESS EMAILS
    # ========================================================

    for email_id in email_ids:

        try:

            # ------------------------------------------------
            # FETCH EMAIL
            # ------------------------------------------------

            status, data = mail.fetch(
                email_id,
                "(RFC822)"
            )


            if status != "OK":

                print(
                    "\n⚠ Could not fetch email:",
                    email_id
                )

                continue


            raw_email = data[0][1]


            message = (
                email.message_from_bytes(
                    raw_email
                )
            )


            # ------------------------------------------------
            # EMAIL MESSAGE ID
            # ------------------------------------------------

            message_id = message.get(
                "Message-ID"
            )


            if message_id:

                message_id = (
                    message_id.strip()
                )


                if message_id in processed_message_ids:

                    print(
                        "\n⏭️ Email already "
                        "processed in this run."
                    )

                    continue


                processed_message_ids.add(
                    message_id
                )


            # ------------------------------------------------
            # EMAIL DETAILS
            # ------------------------------------------------

            sender = message.get(
                "From",
                "Unknown"
            )


            subject = decode_subject(
                message.get(
                    "Subject",
                    "No Subject"
                )
            )


            date = message.get(
                "Date",
                "Unknown"
            )


            print(
                "\n" + "-" * 70
            )


            print(
                "From    :",
                sender
            )


            print(
                "Subject :",
                subject
            )


            print(
                "Date    :",
                date
            )


            # =================================================
            # PROCESS ATTACHMENTS
            # =================================================

            found_supported_attachment = False


            for part in message.walk():

                # ------------------------------------------------
                # Ignore multipart containers
                # ------------------------------------------------

                if (
                    part.get_content_maintype()
                    == "multipart"
                ):

                    continue


                # ------------------------------------------------
                # GET FILENAME
                # ------------------------------------------------

                filename = (
                    part.get_filename()
                )


                content_type = (
                    part.get_content_type()
                )


                # ------------------------------------------------
                # Some inline files may not have filename
                # ------------------------------------------------

                if not filename:

                    print(
                        "⚠ File without filename:",
                        content_type
                    )

                    continue


                filename = safe_filename(
                    filename
                )


                # ------------------------------------------------
                # GET EXTENSION
                # ------------------------------------------------

                extension = (
                    os.path.splitext(
                        filename
                    )[1]
                    .lower()
                )


                print(
                    f"📎 Found attachment: "
                    f"{filename} "
                    f"({content_type})"
                )


                # ------------------------------------------------
                # CHECK SUPPORTED FORMAT
                # ------------------------------------------------

                if (
                    extension
                    not in SUPPORTED_EXTENSIONS
                ):

                    print(
                        "⚠ Skipping unsupported "
                        "attachment:",
                        filename
                    )

                    unsupported_count += 1

                    continue


                found_supported_attachment = True


                # ------------------------------------------------
                # DECODE ATTACHMENT
                # ------------------------------------------------

                payload = part.get_payload(
                    decode=True
                )


                if not payload:

                    print(
                        "⚠ Empty attachment:",
                        filename
                    )

                    empty_count += 1

                    continue


                # ------------------------------------------------
                # CHECK DUPLICATE
                # ------------------------------------------------

                if attachment_already_exists(
                    filename,
                    payload
                ):

                    print(
                        "⏭️ Attachment already "
                        "exists:",
                        filename
                    )

                    duplicate_count += 1

                    continue


                # ------------------------------------------------
                # GET UNIQUE PATH
                # ------------------------------------------------

                file_path = (
                    get_unique_file_path(
                        filename
                    )
                )


                # ------------------------------------------------
                # SAVE ATTACHMENT
                # ------------------------------------------------

                try:

                    with open(
                        file_path,
                        "wb"
                    ) as file:

                        file.write(
                            payload
                        )


                    saved_count += 1


                    print(
                        "✅ Attachment saved:",
                        os.path.basename(
                            file_path
                        )
                    )


                except Exception as e:

                    print(
                        "❌ Could not save:",
                        filename
                    )

                    print(e)


            if not found_supported_attachment:

                print(
                    "No supported attachments."
                )


        except Exception as e:

            print(
                "\n⚠ Error processing email:"
            )

            print(e)


    # ========================================================
    # LOGOUT
    # ========================================================

    try:

        mail.logout()

    except Exception:

        pass


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )


    print(
        "ATTACHMENT EXTRACTION COMPLETED"
    )


    print(
        "=" * 70
    )


    print(
        f"✅ New attachments saved : "
        f"{saved_count}"
    )


    print(
        f"⏭️ Duplicates skipped    : "
        f"{duplicate_count}"
    )


    print(
        f"⚠ Unsupported skipped    : "
        f"{unsupported_count}"
    )


    print(
        f"⚠ Empty attachments      : "
        f"{empty_count}"
    )


    print(
        f"\n📁 Attachment folder:"
    )


    print(
        ATTACHMENT_FOLDER
    )


    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_attachments()