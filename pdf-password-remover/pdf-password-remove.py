import PyPDF2

def remove_password(input_pdf, output_pdf):
    # Open the PDF file
    with open(input_pdf, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Check if the PDF is encrypted
        if reader.is_encrypted:
            password = input("Enter the password for the PDF: ")
            reader.decrypt(password)

        # Create a writer object to write changes to a new PDF
        writer = PyPDF2.PdfWriter()

        # Copy pages from the input PDF to the writer object
        for page_num in range(len(reader.pages)):
            writer.add_page(reader.pages[page_num])

        # Write the new PDF without a password
        with open(output_pdf, 'wb') as output_file:
            writer.write(output_file)

        print("Password removed successfully. New PDF saved as", output_pdf)

if __name__ == "__main__":
    input_pdf = input("Enter the path to the input PDF file: ").strip('"')
    output_pdf = input("Enter the path to save the output PDF file (without password): ")

    remove_password(input_pdf, output_pdf)
