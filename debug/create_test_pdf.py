from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Security Requirements: Best Practices & Examples', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# --- CONTENT ---

# Intro
pdf.chapter_title("1. Introduction")
pdf.chapter_body(
    "This document outlines common security requirement pitfalls. "
    "It contrasts vague, unverifiable requirements ('Bad') with specific, testable ones ('Good'). "
    "Use this to guide the refinement of security specifications for the Employee Portal App."
)

# Authentication
pdf.chapter_title("2. Authentication Requirements")
pdf.chapter_body(
    "BAD EXAMPLE: \"The system must have strong passwords.\"\n"
    "WHY IT FAILS: 'Strong' is subjective. Developers might interpret this as just 6 characters.\n\n"
    "GOOD EXAMPLE: \"The system must enforce a password policy requiring a minimum length of 12 characters, "
    "containing at least one uppercase letter, one lowercase letter, one number, and one special character. "
    "Passwords must be hashed using Argon2id or bcrypt.\""
)

# Data Protection
pdf.chapter_title("3. Data Protection (Encryption)")
pdf.chapter_body(
    "BAD EXAMPLE: \"The application must be secure and encrypt data.\"\n"
    "WHY IT FAILS: Does not specify which data, which state (at rest/in transit), or which algorithm.\n\n"
    "GOOD EXAMPLE: \"All Sensitive PII (Personally Identifiable Information) stored in the database "
    "must be encrypted at rest using AES-256. All data in transit must use TLS 1.3 only.\""
)

# Access Control
pdf.chapter_title("4. Access Control")
pdf.chapter_body(
    "BAD EXAMPLE: \"Only admins can access settings.\"\n"
    "WHY IT FAILS: Ambiguous. Who is an admin? How is it enforced?\n\n"
    "GOOD EXAMPLE: \"The system must implement Role-Based Access Control (RBAC). "
    "Only users assigned the 'System_Administrator' role are permitted to access the '/settings' endpoint. "
    "Access attempts by unauthorized roles must result in a HTTP 403 Forbidden response.\""
)

# Logging
pdf.chapter_title("5. Auditing and Logging")
pdf.chapter_body(
    "BAD EXAMPLE: \"The system should log errors.\"\n"
    "WHY IT FAILS: 'Should' is not mandatory. Doesn't specify what to log.\n\n"
    "GOOD EXAMPLE: \"The system MUST log all failed authentication attempts, privilege escalation events, "
    "and access to sensitive data. Logs must include: Timestamp (UTC), Source IP, User ID, and Event Type. "
    "Logs must not contain plain-text passwords.\""
)

# Save
output_filename = "Security_Requirements_Test.pdf"
pdf.output(output_filename)
print(f"PDF Generated successfully: {output_filename}")