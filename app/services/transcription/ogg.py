# https://datatracker.ietf.org/doc/html/rfc3533
# Format of the Ogg page header:

#  0                   1                   2                   3
#  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1| Byte
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# | capture_pattern: Magic number for page start "OggS"           | 0-3
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# | version       | header_type   | granule_position              | 4-7
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                                                               | 8-11
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                               | bitstream_serial_number       | 12-15
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                               | page_sequence_number          | 16-19
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                               | CRC_checksum                  | 20-23
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                               |page_segments  | segment_table | 24-27
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# | ...                                                           | 28-
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# Explanation of the fields in the page header:

# 1. capture_pattern: a 4-byte field indicating the start of a page. 
#    It contains the magic numbers:
#       0x4f 'O'
#       0x67 'g'
#       0x67 'g'
#       0x53 'S'
#    This helps a decoder find page boundaries and regain synchronization after parsing a corrupted stream. 
#    After finding the capture pattern, the decoder verifies page sync and integrity by computing and comparing the checksum.

# 2. stream_structure_version: a 1-byte field indicating the Ogg file format version used in this stream (this document specifies version 0).

# 3. header_type_flag: a 1-byte field specifying the type of this page.
#    - Continuation Flag (0x01): Bit 0. If set (1), it indicates that this page continues a packet from the previous page. 
#      If not set (0), the first packet on this page is the start of a new packet.
#    - Beginning of Stream Flag (0x02): Bit 1. If set (1), it signals the start of a stream. 
#      Typically set only on the first page of an Ogg stream.
#    - End of Stream Flag (0x04): Bit 2. If set (1), it marks the end of the stream. 
#      Set only on the last page of a logical stream.
#    - Reserved Bits (0x08 to 0x80): Bits 3 to 7 are reserved for future use. Typically set to 0.

# 4. granule_position: an 8-byte field with position information, e.g., total number of PCM samples or video pages encoded after this page. 
#    A value of -1 indicates that no packets finish on this page.

# 5. bitstream_serial_number: a 4-byte field containing the unique serial number identifying the logical bitstream.

# 6. page_sequence_number: a 4-byte field with the page's sequence number, helping the decoder identify page loss. 
#    Increments with each page in each logical bitstream.

# 7. CRC_checksum: a 4-byte field with a 32-bit CRC checksum of the page, including header with zero CRC field and page content. 
#    The generator polynomial is 0x04c11db7.

# 8. number_page_segments: a 1-byte field indicating the number of segment entries in the segment table.

# 9. segment_table: a series of bytes equal to the number_page_segments, containing the lacing values of all segments in this page.

# Total header size in bytes: header_size = number_page_segments + 27 [Byte]
# Total page size in bytes: page_size = header_size + sum(lacing_values: 1..number_page_segments) [Byte]


from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class OggS_Page:
    raw_data: bytes                                 # Entire page data with header and segments
    endianness: Literal["little", "big"] = "little" # Endianness of the data
    capture_pattern: bytes = field(init=False)      # Magic number for page start "OggS"
    version: int = field(init=False)                # Ogg file format version used in this stream
    header_type: int = field(init=False)            # Type of this page
    granule_position: int = field(init=False)       # Position information, e.g., total number of PCM samples or video pages encoded after this page
    serial_number: int = field(init=False)          # Unique serial number identifying the logical bitstream
    page_sequence_number: int = field(init=False)   # Page's sequence number
    CRC_checksum: int = field(init=False)           # 32-bit CRC checksum of the page
    segment_count: int = field(init=False)          # Number of segment entries in the segment table
    segment_table: bytes = field(init=False)        # Lacing values of all segments in this page
    page_size: int = field(init=False)              # Total page size in bytes
    duration: float = field(default=0.0)            # Duration of the page in seconds
    data: bytes = field(init=False)                 # Page data

    def __post_init__(self) -> None:
        """Extract the header information from the raw data."""
        self._from_bytes()

    def _from_bytes(self) -> None:
        """Create an OggS_Page object from raw bytes."""
        page_data: bytes = self.raw_data
        endianness: Literal['little', 'big'] = self.endianness
        if len(page_data) < 27 or page_data[0:4] != b'OggS':
            raise ValueError("Invalid OggS page data.")

        # Extract header information
        self.capture_pattern = page_data[0:4]
        self.version = page_data[4]
        self.header_type = page_data[5]
        self.granule_position = int.from_bytes(page_data[6:14], endianness)
        self.serial_number = int.from_bytes(page_data[14:18], endianness)
        self.page_sequence_number = int.from_bytes(page_data[18:22], endianness)
        self.CRC_checksum = int.from_bytes(page_data[22:26], endianness)
        self.segment_count = page_data[26]

        # Extract the segment table and calculate the total page size
        self.segment_table = page_data[27:27 + self.segment_count]
        self.page_size = 27 + len(self.segment_table) + sum(self.segment_table)

        # Extract page data
        self.data = page_data[27 + self.segment_count:self.page_size]

    def __repr__(self) -> str:
        return (f"OggS_Page(Header: {{'capture_pattern': {str(self.capture_pattern)}, "
                f"'version': {self.version}, "
                f"'header_type': {self.header_type}, "
                f"'granule_position': {self.granule_position}, "
                f"'serial_number': {self.serial_number}, "
                f"'page_sequence_number': {self.page_sequence_number}, "
                f"'CRC_checksum': {str(self.CRC_checksum)}, "
                f"'segment_count': {self.segment_count}}}, "
                f"Raw data length: {len(self.raw_data)}, "
                f"Data length: {len(self.data)})")


#       Identification Header
#       0                   1                   2                   3
#       0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |      'O'      |      'p'      |      'u'      |      's'      |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |      'H'      |      'e'      |      'a'      |      'd'      |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |  Version = 1  | Channel Count |           Pre-skip            |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |                     Input Sample Rate (Hz)                    |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |   Output Gain (Q7.8 in dB)    | Mapping Family|               |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+               :
#      |                                                               |
#      :               Optional Channel Mapping Table...               :
#      |                                                               |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
@dataclass
class OPUS_Id_Header:
    page: OggS_Page                                     # OggS page containing the ID header
    version: int = field(init=False)                    # Version number
    channel_count: int = field(init=False)              # Number of audio channels
    pre_skip: int = field(init=False)                   # Pre-skip value
    input_sample_rate: int = field(init=False)          # Input sample rate in Hz
    output_gain: int = field(init=False)                # Output gain in dB
    mapping_family: int = field(init=False)             # Channel mapping family
    optional_mapping_table: bytes = field(init=False)   # Optional channel mapping table

    def __post_init__(self) -> None:
        """Extract the ID Header details from the page data."""
        self._from_page()

    def _from_page(self) -> None:
        """Extract the ID Header details from the page data."""
        data: bytes = self.page.data
        self.version = data[8]
        self.channel_count = data[9]
        self.pre_skip = int.from_bytes(data[10:12], 'little')
        self.input_sample_rate = int.from_bytes(data[12:16], 'little')
        self.output_gain = int.from_bytes(data[16:18], 'little', signed=True)
        self.mapping_family = data[18]
        self.optional_mapping_table = data[19:] if len(data) > 19 else b''


#       Comment Header
#       0                   1                   2                   3
#       0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |      'O'      |      'p'      |      'u'      |      's'      |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |      'T'      |      'a'      |      'g'      |      's'      |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |                     Vendor String Length                      |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |                                                               |
#      :                        Vendor String...                       :
#      |                                                               |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |                   User Comment List Length                    |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |                 User Comment #0 String Length                 |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |                                                               |
#      :                   User Comment #0 String...                   :
#      |                                                               |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      |                 User Comment #1 String Length                 |
#      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#      :                                                               :
@dataclass
class OPUS_Comment_Header:
    pages: List[OggS_Page]                          # OggS pages containing the comment header
    vendor_string_length: int = field(init=False)   # Length of the vendor string
    vendor_string: str = field(init=False)          # Vendor string
    user_comments: List[str] = field(init=False)    # List of user comments

    def __post_init__(self) -> None:
        """Extract the Comment Header details from the pages."""
        self._from_pages()

    def _from_pages(self) -> None:
        """Extract the Comment Header details from the pages."""
        # Concatenate the data from all pages belonging to the comment header
        data: bytes = b''.join(page.data for page in self.pages)

        # Check if data starts with 'OpusTags' (8 bytes)
        if not data.startswith(b'OpusTags'):
            raise ValueError("Invalid Comment Header: Does not start with 'OpusTags'.")

        # Skip the 'OpusTags' string (8 bytes)
        offset: int = 8

        # Read the Vendor String Length (4 bytes, little-endian)
        self.vendor_string_length = int.from_bytes(data[offset:offset + 4], 'little')
        offset += 4

        # Read the Vendor String
        self.vendor_string = data[offset:offset + self.vendor_string_length].decode('utf-8')
        offset += self.vendor_string_length

        # Read the User Comment List Length (4 bytes, little-endian)
        user_comment_list_length: int = int.from_bytes(data[offset:offset + 4], 'little')
        offset += 4

        # Read each User Comment
        self.user_comments = []
        for _ in range(user_comment_list_length):
            # Read the User Comment #N String Length (4 bytes, little-endian)
            if offset + 4 > len(data):
                raise ValueError("Unexpected end of data while reading user comment length.")

            user_comment_length: int = int.from_bytes(data[offset:offset + 4], 'little')
            offset += 4

            # Read the User Comment #N String
            if offset + user_comment_length > len(data):
                raise ValueError("Unexpected end of data while reading user comment.")

            user_comment: str = data[offset:offset + user_comment_length].decode('utf-8')
            self.user_comments.append(user_comment)
            offset += user_comment_length



# https://datatracker.ietf.org/doc/html/rfc7845.html#section-5.2
# Packet Organization in an Ogg Opus stream

# An Ogg Opus stream is organized as follows (see Figure 1 for an example).

#         Page 0         Pages 1 ... n        Pages (n+1) ...
#      +------------+ +---+ +---+ ... +---+ +-----------+ +---------+ +--
#      |            | |   | |   |     |   | |           | |         | |
#      |+----------+| |+-----------------+| |+-------------------+ +-----
#      |||ID Header|| ||  Comment Header || ||Audio Data Packet 1| | ...
#      |+----------+| |+-----------------+| |+-------------------+ +-----
#      |            | |   | |   |     |   | |           | |         | |
#      +------------+ +---+ +---+ ... +---+ +-----------+ +---------+ +--
#      ^      ^                           ^
#      |      |                           |
#      |      |                           Mandatory Page Break
#      |      |
#      |      ID header is contained on a single page
#      |
#      'Beginning Of Stream'

#     Figure 1: Example Packet Organization for a Logical Ogg Opus Stream
@dataclass
class Ogg_OPUS_Audio:
    ogg_data: bytes
    pages: List[OggS_Page] = field(init=False)
    id_header: Optional[OPUS_Id_Header] = field(init=False)
    comment_header: Optional[OPUS_Comment_Header] = field(init=False)
    duration: float = field(default=0.0)    # Duration of the audio data in seconds

    def __post_init__(self) -> None:
        """Initialize the Ogg_OPUS_Audio object and split the ogg_data into pages."""
        self.pages = self._split_ogg_data_into_pages(self.ogg_data)
        self.id_header = self._extract_id_header()
        self.comment_header = self._extract_comment_header()
        self._calculate_page_duration()

    def _split_ogg_data_into_pages(self, ogg_data: bytes) -> List[OggS_Page]:
        """Split the Ogg data into individual pages."""
        pages: List[OggS_Page] = []
        offset: int = 0

        while offset < len(ogg_data):
            # Read the first 27 bytes of the header
            header: bytes = ogg_data[offset:offset + 27]
            if len(header) < 27 or header[0:4] != b'OggS':
                break  # End if no valid header is found

            # Read the number of segments
            page_segments: int = header[26]
            segment_table: bytes = ogg_data[offset + 27:offset + 27 + page_segments]

            # Calculate the total page size
            page_size: int = 27 + page_segments + sum(segment_table)

            # Extract the entire page
            page: bytes = ogg_data[offset:offset + page_size]
            ogg_page: OggS_Page = OggS_Page(page)
            pages.append(ogg_page)

            # Update offset to the next page
            offset += page_size

        # Sort pages into the correct order by page_sequence_number
        sorted_pages: List[OggS_Page] = sorted(pages, key=lambda page: page.page_sequence_number)
        return sorted_pages

    def __repr__(self) -> str:
        return (f"Ogg_OPUS_Audio(Total Pages: {len(self.pages)}, "
                f"Data length: {len(self.ogg_data)})")

    def _extract_id_header(self) -> Optional[OPUS_Id_Header]:
        """
        Extract the ID Header page from the Ogg Opus stream and return the page and an OPUS_Id_Header object.
        """
        for page in self.pages:
            if page.data.startswith(b'OpusHead'):
                # Extract the ID Header details
                opus_id_header: OPUS_Id_Header = OPUS_Id_Header(page)
                return opus_id_header
        return None

    def _extract_comment_header(self) -> Optional[OPUS_Comment_Header]:
        """
        Extract the Comment Header pages from the Ogg Opus stream and return the pages and a list of OPUS_Comment_Header objects.
        """
        comment_header_started: bool = False
        comment_header_completed: bool = False
        comment_header_pages: List[OggS_Page] = []

        for page in self.pages:
            if not comment_header_started:
                # Start looking for the Comment Header at page sequence number 1
                if page.page_sequence_number == 1 and page.data.startswith(b'OpusTags'):
                    comment_header_started = True
                    comment_header_pages.append(page)
            else:
                # Continue collecting pages until a non-continuation packet is found
                if page.header_type & 0x01 == 0:  # Checking the 'continued packet' flag
                    comment_header_pages.append(page)
                    comment_header_completed = True
                    break
                else:
                    comment_header_pages.append(page)

        if not comment_header_completed:
            return None

        opus_comment_header: OPUS_Comment_Header = OPUS_Comment_Header(comment_header_pages)
        return opus_comment_header

    def _calculate_page_duration(self) -> None:
        """Calculate the duration of each page in the audio data."""
        pages = self.pages
        if not self.id_header or not self.comment_header:
            return
        id_header_page = self.id_header.page
        comment_header_pages = self.comment_header.pages

        sample_rate = self.id_header.input_sample_rate

        # Remove id_header_page and comment_header_pages from pages
        # pages = [page for page in pages if page != id_header_page and page not in comment_header_pages]

        complete_duration = 0.0
        previous_granule_position: int = pages[0].granule_position
        for page in pages[1:]:
            current_granule_position = page.granule_position
            samples = current_granule_position - previous_granule_position
            duration = samples / sample_rate
            page.duration = duration
            complete_duration += duration

            previous_granule_position = current_granule_position

        self.duration = complete_duration


def calculate_page_duration(current_granule_position: int, previous_granule_position: Optional[int], sample_rate: int = 48000) -> float:
    if previous_granule_position is None:
        return 0.0  # Default value for the first frame
    samples = current_granule_position - previous_granule_position
    duration = samples / sample_rate
    return duration



def __main__() -> None:
    file_path: str = 'audio/bbb.ogg'

    ogg_data: bytes = b""
    try:
        with open(file_path, "rb") as file:
            ogg_data = file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        exit(1)
    except IOError as e:
        print(f"An error occurred while reading {file_path}: {e}")
        exit(1)

    audio: Ogg_OPUS_Audio = Ogg_OPUS_Audio(ogg_data)

    # Print the total number of pages and some details about each page
    print(f"Total pages: {len(audio.pages)}")

    # OPUS
    id_header: Optional[OPUS_Id_Header] = audio.id_header
    if id_header is not None:
        id_header_page: OggS_Page = id_header.page
        print("ID Header Page:")
        print(f"  Capture Pattern: {id_header_page.capture_pattern.decode()}")
        print(f"  Version: {id_header_page.version}")
        print(f"  Header Type: {id_header_page.header_type}")
        print(f"  Granule Position: {id_header_page.granule_position}")
        print(f"  Serial Number: {id_header_page.serial_number}")
        print(f"  Page Sequence Number: {id_header_page.page_sequence_number}")
        print(f"  CRC Checksum: {id_header_page.CRC_checksum}")
        print(f"  Segment Count: {id_header_page.segment_count}")
        print(f"  Data Length: {len(id_header_page.data)} bytes")
        print("OPUS ID Header:")
        print(f"  Version: {id_header.version}")
        print(f"  Channel Count: {id_header.channel_count}")
        print(f"  Pre-skip: {id_header.pre_skip}")
        print(f"  Input Sample Rate: {id_header.input_sample_rate}")
        print(f"  Output Gain: {id_header.output_gain}")
        print(f"  Mapping Family: {id_header.mapping_family}")
        print(f"  Optional Mapping Table: {id_header.optional_mapping_table.decode()}")
    else:
        print("No ID header page found.")
        exit(1)
    print(f"\n")
    comment_header: Optional[OPUS_Comment_Header] = audio.comment_header
    if comment_header is not None:
        comment_header_pages: List[OggS_Page] = comment_header.pages
        print(f"Comment Header Pages: {len(comment_header_pages)}")
        for index, page in enumerate(comment_header_pages):
            print(f"Comment Header Page {index + 1}:")
            print(f"  Capture Pattern: {page.capture_pattern.decode()}")
            print(f"  Version: {page.version}")
            print(f"  Header Type: {page.header_type}")
            print(f"  Granule Position: {page.granule_position}")
            print(f"  Serial Number: {page.serial_number}")
            print(f"  Page Sequence Number: {page.page_sequence_number}")
            print(f"  CRC Checksum: {page.CRC_checksum}")
            print(f"  Segment Count: {page.segment_count}")
            print(f"  Data Length: {len(page.data)} bytes") 
        print("OPUS Comment Header:")
        print(f"  Vendor String Length: {comment_header.vendor_string_length}")
        print(f"  Vendor String: {comment_header.vendor_string}")
        print(f"  User Comment List Length: {len(comment_header.user_comments)}")
        for index, user_comment in enumerate(comment_header.user_comments):
            print(f"  User Comment {index + 1}: {user_comment}")

    else:
        print("No Comment header pages found.")
        exit(1)
    print(f"\n")
    # print duration hours, minutes, seconds
    print(f"Duration: {int(audio.duration / 3600)}:{int((audio.duration % 3600) / 60)}:{int(audio.duration % 60)}")

if __name__ == "__main__":
    __main__()
