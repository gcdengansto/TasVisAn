import re
from typing import List, Tuple

class TaipanCommandValidator:
    # Compile regex patterns for efficiency
    NUMBER_PATTERN = r'-?\d+\.?\d*'
    MOTOR_PATTERN = r'(qh|qk|ql|en)'
    COUNTER_PATTERN = r'(monitor|time)'
    
    # Pattern for motor-value pairs (e.g., "qh 0 0" or "qh 1")
    MOTOR_VALUE_PAIR = rf'{MOTOR_PATTERN}\s+{NUMBER_PATTERN}(?:\s+{NUMBER_PATTERN})?'
    
    # Complete command patterns
    MSCAN_PATTERN = re.compile(
        rf'^mscan\s+{MOTOR_VALUE_PAIR}\s+{MOTOR_VALUE_PAIR}\s+{MOTOR_VALUE_PAIR}\s+{MOTOR_VALUE_PAIR}\s+'
        rf'{NUMBER_PATTERN}\s+{COUNTER_PATTERN}\s+{NUMBER_PATTERN}$'
    )
    
    DRIVE_PATTERN = re.compile(
        rf'^drive\s+({MOTOR_PATTERN}\s+{NUMBER_PATTERN}\s*)+$'
    )
    
    RUNSCAN_PATTERN = re.compile(
        rf'^runscan\s+{MOTOR_PATTERN}\s+{NUMBER_PATTERN}\s+{NUMBER_PATTERN}\s+'
        rf'{NUMBER_PATTERN}\s+{COUNTER_PATTERN}\s+{NUMBER_PATTERN}$'
    )
    
    # New pattern for title command - accepts any non-empty string after 'title'
    TITLE_PATTERN = re.compile(r'^title\s+.+$')

    @classmethod
    def validate_command(cls, command: str) -> Tuple[bool, str]:
        """
        Validates the syntax of scanning commands.
        Returns (is_valid, error_message)
        """
        # Remove extra whitespace but preserve newlines
        command = '\n'.join(line.strip() for line in command.split('\n'))
        
        # Check command type
        if command.startswith('mscan'):
            return cls.validate_mscan(command)
        elif command.startswith('drive'):
            return cls.validate_drive_command(command)
        elif command.startswith('title'):
            return cls.validate_title(command)
        
        return False, "Command must start with 'mscan', 'drive', or 'title'"

    @classmethod
    def validate_title(cls, command: str) -> Tuple[bool, str]:
        """Validates title command syntax"""
        if not cls.TITLE_PATTERN.match(command):
            return False, "Title command must be followed by a non-empty string"
        return True, "Valid title command"

    @classmethod
    def validate_mscan(cls, command: str) -> Tuple[bool, str]:
        """Validates mscan command syntax using regex"""
        if not cls.MSCAN_PATTERN.match(command):
            return False, "Invalid mscan command format"
        
        # Extract and validate numeric values
        numbers = re.findall(cls.NUMBER_PATTERN, command)
        
        # Validate points (second to last number)
        try:
            points = int(float(numbers[-2]))
            if points <= 0:
                return False, "Number of points must be positive"
        except (IndexError, ValueError):
            return False, "Invalid number of points"
        
        # Validate counter value (last number)
        try:
            counter_value = float(numbers[-1])
            if counter_value <= 0:
                return False, "Counter value must be positive"
        except (IndexError, ValueError):
            return False, "Invalid counter value"
        
        return True, "Valid mscan command"

    @classmethod
    def validate_drive_command(cls, command: str) -> Tuple[bool, str]:
        """Validates drive command syntax (single or with runscan)"""
        lines = command.split('\n')
        
        # If single drive command
        if len(lines) == 1:
            return cls.validate_single_drive(lines[0])
        
        # If drive + runscan pair
        if len(lines) == 2:
            return cls.validate_drive_runscan_pair(lines[0], lines[1])
        
        return False, "Invalid number of lines for drive command"

    @classmethod
    def validate_single_drive(cls, drive_cmd: str) -> Tuple[bool, str]:
        """Validates a single drive command using regex"""
        drive_cmd = drive_cmd.strip()
        if not cls.DRIVE_PATTERN.match(drive_cmd):
            return False, "Invalid drive command format"
        
        # Additional validation: check for correct number of values
        parts = drive_cmd.split()
        if (len(parts) - 1) % 2 != 0:  # Subtract 1 for 'drive' command
            return False, "Drive command must have pairs of motor and position values"
        
        return True, "Valid drive command"

    @classmethod
    def validate_drive_runscan_pair(cls, drive_cmd: str, runscan_cmd: str) -> Tuple[bool, str]:
        """Validates a drive command followed by a runscan command"""
        # First validate drive command
        drive_valid, drive_message = cls.validate_single_drive(drive_cmd)
        if not drive_valid:
            return False, drive_message
        
        # Validate runscan command
        runscan_cmd = runscan_cmd.strip()
        if not cls.RUNSCAN_PATTERN.match(runscan_cmd):
            return False, "Invalid runscan command format"
        
        # Extract and validate numeric values from runscan
        numbers = re.findall(cls.NUMBER_PATTERN, runscan_cmd)
        
        # Validate points (should be integer)
        try:
            points = int(float(numbers[2]))  # Third number is points
            if points <= 0:
                return False, "Number of points must be positive"
        except (IndexError, ValueError):
            return False, "Invalid number of points in runscan"
        
        # Validate counter value
        try:
            counter_value = float(numbers[3])  # Fourth number is counter value
            if counter_value <= 0:
                return False, "Counter value must be positive"
        except (IndexError, ValueError):
            return False, "Invalid counter value in runscan"
        
        return True, "Valid drive/runscan command"




class SikaCommandValidator:
    def __init__(self):
        self.valid_motors = ['h', 'k', 'l', 'e']
        
    def check_scan_command(self, parts: List[str]) -> bool:
        if len(parts) < 2:
            return False, "The command is too short!"
        command = " ".join(parts[0:])
        if parts[1].startswith('n'):
            # Handle cases like "scan n=100" or "scan n = 100"
            if parts[1] == 'n':
                if len(parts) == 3:
                    return parts[2].isdigit(), "The scan n value command."
                elif len(parts) == 4 and parts[2] == '=':
                    return parts[3].isdigit(), "The scan n = value command."
            elif parts[1]=='n=':
                return  parts[2].isdigit(), "The scan n= value  command."
            elif parts[1].startswith('n='):
                value = parts[1][2:]  # Get everything after 'n='
                return (len(parts) == 2 and value.isdigit() ) , "The scan n=value  command."
            
        else:
                    
            motor_pattern = r'([hkleHKLE])\s+(-?\d*\.?\d+)(?:\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+))?'
        
            # Remove 'scan' from the command and strip whitespace
            #command_params = command.split(maxsplit=1)[1].strip()
            command_params = " ".join(parts[1:])
            # Find all motor specifications in the command
            motors = re.findall(motor_pattern, command_params)
            
            if len(motors) == 0:  #not motors: 
                print("Error: no motor was found in the command.")
                return False, "Error: no motor was found in the command."
                
            # Check if we found exactly 4 motors
            if len(motors) != 4:
                print("Error: incorrect number of motors found in the command.")
                return False,"Error: incorrect number of motors found in the command."
                
            # Keep track of motors we've seen to check for duplicates
            seen_motors = set()
            has_scanning_motor = False
            
            # Validate each motor specification
            for motor_match in motors:
                motor_name = motor_match[0].lower()
                
                # Check for duplicate motors
                if motor_name in seen_motors:
                    print("Error:duplicated motor name:{}".format(motor_name))
                    return False, "Error:duplicated motor name:{}".format(motor_name)
                seen_motors.add(motor_name)
                
                # Check if motor name is valid
                if motor_name not in {'h', 'k', 'l', 'e'}:
                    print("wrong motor name:{}".format(motor_name))
                    return False, "wrong motor name:{}".format(motor_name)
                    
                # Check parameter consistency
                # If we have a second number (motor_match[2]), we must have a third number (motor_match[3])
                has_second = bool(motor_match[2].strip())
                has_third = bool(motor_match[3].strip())
                if has_second != has_third:
                    return False, "the scanning motor needs three parameters"
            
                # Check if this is a scanning motor
                if has_second and has_third:
                    has_scanning_motor = True

            
            if not has_scanning_motor:
                print("ERROR: no scanning motor!")
                return False, "ERROR: no scanning motor!"
            
            # Check if the entire string was consumed (no extra parameters)
            reconstructed_command = 'scan'
            for motor in motors:
                reconstructed_command += f' {motor[0]}'
                reconstructed_command += f' {motor[1]}'
                if motor[2].strip() and motor[3].strip():  # If scanning motor
                    reconstructed_command += f' {motor[2]} {motor[3]}'
            
            if reconstructed_command.strip() != command.strip():
                return False, "ERROR: The scan command contain errors!"
                
            return True, "The current scan command is valid."



    def check_preset_command(self, parts: List[str]) -> bool:
        if len(parts) != 3:
            return False, "The preset command is incorrect."
        return (parts[1] in ['time', 'count']) and parts[2].isdigit(), "The preset command"
    
    def check_drive_command(self, parts: List[str]) -> bool:
        if len(parts) < 3:
            return False, "The drive command does not have enough parameters"
        
        i = 1
        while i < len(parts)-1:
            if parts[i] not in self.valid_motors:
                return False, "The motor has a wrong name."
            if not self.is_number(parts[i+1]):
                return False, "The motor position is not numeric."
            i += 2
        return i == len(parts)
    
    def check_scantitle_command(self, line: str) -> bool:
        pattern = r'^scantitle\s+(["\'])(.*?)\1$'
        return bool(re.match(pattern, line.strip())), "scantitle command"
    
    def is_number(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def validate_command(self, command: str) -> bool:
        """Validate a single command string"""
        parts = command.strip().split()
        if not parts:
            return False, "The command has no parameter."
            
        command_type = parts[0].lower()
        if command_type == 'scan':
            return self.check_scan_command(parts)
        elif command_type == 'preset':
            return self.check_preset_command(parts)
        elif command_type == 'drive':
            return self.check_drive_command(parts)
        elif command_type == 'scantitle':
            return self.check_scantitle_command(command)
        return False, "The command unknown."