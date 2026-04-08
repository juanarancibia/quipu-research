#!/usr/bin/env python3
"""
WhatsApp Chat Parser for Quipu Bot Interactions
Parses WhatsApp chat exports into structured JSON format with date filtering.
"""

import re
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class WhatsAppChatParser:
    """Parser for WhatsApp chat exports to extract Quipu bot interactions."""
    
    # WhatsApp message pattern (format: [DD/MM/YY, HH:MM:SS])
    MESSAGE_PATTERN = r'^[\u200e\s]*\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}:\d{2})\] ([^:]+): (.*)$'
    
    def __init__(self, chat_file_path: str):
        """Initialize parser with chat file path."""
        self.chat_file_path = chat_file_path
        self.conversations: List[Dict] = []
        
    def parse_timestamp(self, date_str: str, time_str: str) -> datetime:
        """Parse WhatsApp timestamp to datetime object."""
        # Format: DD/MM/YY, HH:MM:SS
        datetime_str = f"{date_str} {time_str}"
        return datetime.strptime(datetime_str, "%d/%m/%y %H:%M:%S")
    
    def read_messages(self) -> List[Tuple[datetime, str, str]]:
        """Read and parse all messages from the chat file."""
        messages = []
        current_message = None
        
        with open(self.chat_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                
                # Check if line starts with a timestamp (new message)
                match = re.match(self.MESSAGE_PATTERN, line)
                if match:
                    # Save previous message if exists
                    if current_message:
                        messages.append(current_message)
                    
                    date_str, time_str, sender, content = match.groups()
                    timestamp = self.parse_timestamp(date_str, time_str)
                    current_message = (timestamp, sender, content)
                else:
                    # Continuation of previous message (multiline)
                    if current_message:
                        timestamp, sender, content = current_message
                        current_message = (timestamp, sender, content + '\n' + line)
            
            # Don't forget the last message
            if current_message:
                messages.append(current_message)
        
        return messages
    
    def extract_bot_response_details(self, content: str) -> Optional[Dict]:
        """Extract structured data from bot responses."""
        # Check for expense/income confirmation
        if "✅ Guardado correctamente" in content or "✅ *Guardado correctamente*" in content:
            details = {}
            
            # Extract type (Gasto/Ingreso)
            if "📤 *Gasto*" in content:
                details['type'] = 'expense'
            elif "📥 *Ingreso*" in content:
                details['type'] = 'income'
            else:
                return None
            
            # Extract description
            desc_match = re.search(r'📝 \*Descripción:\* (.+)', content)
            if desc_match:
                details['description'] = desc_match.group(1).strip()
            
            # Extract amount
            amount_match = re.search(r'💰 \*Monto:\* ([\d.,]+) (\w+)', content)
            if amount_match:
                amount_str = amount_match.group(1).replace('.', '').replace(',', '.')
                details['amount'] = float(amount_str)
                details['currency'] = amount_match.group(2)
            
            # Extract category
            category_match = re.search(r'🏷️ \*Categoría:\* (.+)', content)
            if category_match:
                details['category'] = category_match.group(1).strip()
            
            # Extract date
            date_match = re.search(r'🗓️ \*Fecha:\* (.+)', content)
            if date_match:
                details['bot_recorded_date'] = date_match.group(1).strip()
            
            return details
        
        # Check for bot confirmation request (with buttons)
        elif ("✅ Confirmar" in content and "❌ Cancelar" in content):
            details = {}
            
            # Extract type (Gasto/Ingreso)
            if "📤 *Gasto*" in content:
                details['type'] = 'expense'
            elif "📥 *Ingreso*" in content:
                details['type'] = 'income'
            else:
                return None
            
            details['status'] = 'pending_confirmation'
            
            # Extract description
            desc_match = re.search(r'📝 \*Descripción:\* (.+)', content)
            if desc_match:
                details['description'] = desc_match.group(1).strip()
            
            # Extract amount
            amount_match = re.search(r'💰 \*Monto:\* ([\d.,]+) (\w+)', content)
            if amount_match:
                amount_str = amount_match.group(1).replace('.', '').replace(',', '.')
                details['amount'] = float(amount_str)
                details['currency'] = amount_match.group(2)
            
            # Extract category
            category_match = re.search(r'🏷️ \*Categoría:\* (.+)', content)
            if category_match:
                details['category'] = category_match.group(1).strip()
            
            # Extract date
            date_match = re.search(r'🗓️ \*Fecha:\* (.+)', content)
            if date_match:
                details['bot_recorded_date'] = date_match.group(1).strip()
            
            return details
        
        # Check for error messages
        elif "❌" in content:
            return {
                'type': 'error',
                'message': content.strip()
            }
        
        # Check for account linking success
        elif "✅ *¡Cuenta vinculada exitosamente!*" in content:
            return {
                'type': 'account_linked',
                'message': content.strip()
            }
        
        # Check for welcome message
        elif "👋 *¡Bienvenido/a a Quipu!*" in content:
            return {
                'type': 'welcome',
                'message': content.strip()
            }
        
        return None
    
    def build_conversations(self, messages: List[Tuple[datetime, str, str]], 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict]:
        """Build conversation pairs (user input -> bot output) with date filtering.
        Only includes conversations where user sends text and bot responds with structured expense/income data."""
        conversations = []
        i = 0
        
        while i < len(messages):
            timestamp, sender, content = messages[i]
            
            # Apply date filtering
            if start_date and timestamp < start_date:
                i += 1
                continue
            if end_date and timestamp > end_date:
                i += 1
                continue
            
            # Skip system messages
            if sender in ["Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them. Learn more.",
                         "This business is now using a secure service from Meta to manage this chat. Tap to learn more."]:
                i += 1
                continue
            
            # User message (not from Quipu)
            if sender != "Quipu":
                # Skip media messages
                if '<Media omitted>' in content:
                    i += 1
                    continue
                
                # Skip confirmation/cancellation buttons
                if content.strip() in ['✅ Confirmar', '❌ Cancelar']:
                    i += 1
                    continue
                
                user_message = content.strip()
                
                # Look for all bot responses with structured data
                j = i + 1
                bot_outputs = []
                
                # Collect all immediate bot responses
                while j < len(messages) and messages[j][1] == "Quipu":
                    bot_timestamp, _, bot_content = messages[j]
                    bot_details = self.extract_bot_response_details(bot_content)
                    
                    # Only include if bot responded with expense or income data
                    if bot_details and bot_details.get('type') in ['expense', 'income']:
                        bot_outputs.append(bot_details)
                    
                    j += 1
                    # Stop if we hit another user message or non-transaction bot response
                    if j < len(messages) and messages[j][1] != "Quipu":
                        break
                
                # Only add conversation if we have at least one structured bot response
                if bot_outputs:
                    conversation = {
                        'input': user_message,
                        'conversation_date': timestamp.strftime('%Y-%m-%d'),
                        'outputs': bot_outputs  # Changed to array
                    }
                    conversations.append(conversation)
                
                i = j
            else:
                i += 1
        
        return conversations
    
    def parse(self, start_date: Optional[str] = None, 
              end_date: Optional[str] = None) -> List[Dict]:
        """
        Parse chat file and extract conversations with optional date filtering.
        
        Args:
            start_date: Start date in format YYYY-MM-DD (inclusive)
            end_date: End date in format YYYY-MM-DD (inclusive)
        
        Returns:
            List of conversation dictionaries
        """
        # Parse date filters
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            # Set to end of day
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        
        # Read messages
        messages = self.read_messages()
        
        # Build conversations
        conversations = self.build_conversations(messages, start_dt, end_dt)
        
        return conversations
    
    def save_to_json(self, output_file: str, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None, indent: int = 2):
        """Parse and save conversations to JSON file."""
        conversations = self.parse(start_date, end_date)
        
        output_data = {
            'metadata': {
                'source_file': self.chat_file_path,
                'total_conversations': len(conversations),
                'start_date_filter': start_date,
                'end_date_filter': end_date,
                'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'note': 'Datos tentativos para análisis manual. Filtrado: solo mensajes de texto con respuesta estructurada (expense/income).'
            },
            'conversations': conversations
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=indent, ensure_ascii=False)
        
        print(f"✅ Parsed {len(conversations)} conversations")
        print(f"📁 Saved to: {output_file}")
        
        return output_data


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Parse WhatsApp chat with Quipu bot into structured JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse entire chat
  python3 parse_whatsapp_chat.py -i files/WhatsApp\ Chat\ with\ Quipu.txt -o output.json
  
  # Parse with date range
  python3 parse_whatsapp_chat.py -i files/WhatsApp\ Chat\ with\ Quipu.txt -o output.json --start-date 2025-06-20 --end-date 2025-06-30
  
  # Parse from specific date onwards
  python3 parse_whatsapp_chat.py -i files/WhatsApp\ Chat\ with\ Quipu.txt -o output.json --start-date 2025-07-01
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to WhatsApp chat export file'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file path'
    )
    
    parser.add_argument(
        '--start-date',
        help='Start date filter (YYYY-MM-DD, inclusive)'
    )
    
    parser.add_argument(
        '--end-date',
        help='End date filter (YYYY-MM-DD, inclusive)'
    )
    
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation level (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Validate date formats
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print("❌ Error: start-date must be in YYYY-MM-DD format")
            return 1
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("❌ Error: end-date must be in YYYY-MM-DD format")
            return 1
    
    # Parse and save
    try:
        parser_obj = WhatsAppChatParser(args.input)
        parser_obj.save_to_json(
            args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            indent=args.indent
        )
        return 0
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
