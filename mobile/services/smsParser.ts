/**
 * SMS Transaction Parser
 * Parses bank/UPI SMS messages to extract transaction details
 */

export interface ParsedTransaction {
  type: 'income' | 'expense' | 'transfer';
  amount: number;
  category: string;
  description: string;
  date: Date;
  recipient?: string;
  merchantName?: string;
  accountNumber?: string;
  balance?: number;
  referenceId?: string;
}

export interface SMSMessage {
  _id: string;
  address: string;
  body: string;
  date: number;
  read: number;
  type: number;
}

// Common bank sender IDs
const BANK_SENDERS = [
  'SBIINB', 'SBIPSG', 'SBISBI', 'SBIBNK',  // SBI
  'HDFCBK', 'HDFCBN', 'HDFC',              // HDFC
  'ICICIB', 'ICICBK', 'ICICI',             // ICICI
  'AXISBK', 'AXIS',                         // Axis
  'KOTAKB', 'KOTAK',                        // Kotak
  'PNBSMS', 'PUNBNK',                       // PNB
  'BOIIND', 'BOBANK',                       // BOI
  'CANBNK', 'CANARA',                       // Canara
  'INDBNK', 'INDIAN',                       // Indian Bank
  'UNIONB', 'UCOBNK',                       // Union/UCO
  'YESBNK', 'YESBK',                        // Yes Bank
  'IABORC', 'IDFCFB',                       // IDFC
  'PAYTMB', 'PYTM',                         // Paytm
  'PHONEPE', 'PHNEPE',                      // PhonePe
  'GPAY', 'GOOGLE',                         // GPay
  'AMAZONP', 'AMZN',                        // Amazon Pay
  'BHARPE', 'BHRPE',                        // BharatPe
  'MOBIKWIK', 'MBKWIK',                     // MobiKwik
  'FREECHARGE', 'FRCHG',                    // Freecharge
  'JIOPAY', 'JIOMNY',                       // Jio
];

// UPI Apps
const UPI_KEYWORDS = ['UPI', 'IMPS', 'NEFT', 'RTGS', 'BHIM'];

// Transaction keywords
const DEBIT_KEYWORDS = [
  'debited', 'debit', 'withdrawn', 'withdrawal', 'spent', 
  'paid', 'purchase', 'payment', 'transferred', 'sent',
  'dr', 'deducted', 'charged'
];

const CREDIT_KEYWORDS = [
  'credited', 'credit', 'received', 'deposit', 'deposited',
  'refund', 'cashback', 'cr', 'added', 'transferred to'
];

// Category detection patterns
const CATEGORY_PATTERNS: { pattern: RegExp; category: string }[] = [
  // Food & Dining
  { pattern: /swiggy|zomato|dominos|pizza|restaurant|food|cafe|hotel|dining|mcdonald|kfc|burger|subway/i, category: 'food' },
  // Shopping
  { pattern: /amazon|flipkart|myntra|ajio|shopping|mall|retail|store|mart|bazaar|market/i, category: 'shopping' },
  // Transport
  { pattern: /uber|ola|rapido|metro|petrol|diesel|fuel|parking|toll|irctc|railway|flight|booking/i, category: 'transport' },
  // Utilities
  { pattern: /electricity|water|gas|bill|recharge|mobile|broadband|internet|dth|postpaid|prepaid/i, category: 'utilities' },
  // Entertainment
  { pattern: /netflix|prime|hotstar|spotify|movie|cinema|pvr|inox|gaming|subscription/i, category: 'entertainment' },
  // Healthcare
  { pattern: /hospital|medical|pharmacy|medicine|doctor|clinic|health|diagnostic|apollo|fortis/i, category: 'healthcare' },
  // Education
  { pattern: /school|college|university|course|tuition|education|exam|fees|udemy|coursera/i, category: 'education' },
  // Groceries
  { pattern: /grocery|bigbasket|grofers|blinkit|zepto|instamart|dmart|reliance fresh|more supermarket/i, category: 'groceries' },
  // Insurance
  { pattern: /insurance|lic|policy|premium|hdfc life|icici pru|sbi life/i, category: 'insurance' },
  // Investment
  { pattern: /mutual fund|mf|sip|zerodha|groww|upstox|investment|stock|share|nse|bse/i, category: 'investment' },
  // Rent
  { pattern: /rent|landlord|housing|apartment|flat|society|maintenance/i, category: 'rent' },
  // ATM
  { pattern: /atm|cash withdrawal|atm withdrawal/i, category: 'cash' },
  // Transfer
  { pattern: /transfer|sent to|paid to|received from/i, category: 'transfer' },
  // Salary
  { pattern: /salary|wages|payroll|income|credited.*salary/i, category: 'salary' },
];

/**
 * Parse amount from SMS text
 */
function parseAmount(text: string): number | null {
  // Match patterns like Rs.1000, Rs 1,000.00, INR 500, ₹1234.56
  const patterns = [
    /(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)/i,
    /(?:amount|amt)(?:\s*:?\s*)(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)/i,
    /([\d,]+(?:\.\d{1,2})?)\s*(?:rs\.?|inr|₹)/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      const amount = parseFloat(match[1].replace(/,/g, ''));
      if (!isNaN(amount) && amount > 0) {
        return amount;
      }
    }
  }

  return null;
}

/**
 * Parse balance from SMS text
 */
function parseBalance(text: string): number | null {
  const patterns = [
    /(?:bal(?:ance)?|avl\.?\s*bal|available)(?:\s*:?\s*)(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)/i,
    /(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:bal(?:ance)?|available)/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      const balance = parseFloat(match[1].replace(/,/g, ''));
      if (!isNaN(balance)) {
        return balance;
      }
    }
  }

  return null;
}

/**
 * Parse account number from SMS
 */
function parseAccountNumber(text: string): string | null {
  // Match patterns like A/c XX1234, Acct *1234, a/c ending 5678
  const patterns = [
    /(?:a\/c|acct?|account)(?:\s*(?:no\.?|number)?)\s*(?:xx|x|\*)*(\d{4,})/i,
    /(?:a\/c|acct?|account)\s*(?:ending|end)\s*(\d{4})/i,
    /\*{2,}(\d{4})/,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      return match[1];
    }
  }

  return null;
}

/**
 * Parse reference ID / transaction ID
 */
function parseReferenceId(text: string): string | null {
  const patterns = [
    /(?:ref(?:erence)?|txn|transaction|utr)(?:\s*(?:no\.?|id|number)?)\s*:?\s*([a-z0-9]+)/i,
    /(?:imps|neft|upi)(?:\s*ref)?\s*:?\s*([a-z0-9]+)/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      return match[1];
    }
  }

  return null;
}

/**
 * Parse merchant/recipient name
 */
function parseMerchant(text: string): string | null {
  const patterns = [
    /(?:to|at|from|@)\s+([a-z0-9\s\-\.]+?)(?:\s+(?:on|ref|upi|via|$))/i,
    /(?:paid to|sent to|received from|transferred to)\s+([a-z0-9\s\-\.]+?)(?:\s+(?:on|ref|$))/i,
    /vpa\s*:?\s*([a-z0-9\.\-@]+)/i,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      return match[1].trim().substring(0, 50);
    }
  }

  return null;
}

/**
 * Detect transaction type (income/expense)
 */
function detectTransactionType(text: string): 'income' | 'expense' | 'transfer' {
  const lowerText = text.toLowerCase();

  // Check for credit keywords first
  for (const keyword of CREDIT_KEYWORDS) {
    if (lowerText.includes(keyword)) {
      // Make sure it's not "credited to someone else"
      if (lowerText.includes('credited to your') || 
          lowerText.includes('credited to a/c') ||
          lowerText.includes('received') ||
          lowerText.includes('refund') ||
          lowerText.includes('cashback')) {
        return 'income';
      }
    }
  }

  // Check for debit keywords
  for (const keyword of DEBIT_KEYWORDS) {
    if (lowerText.includes(keyword)) {
      return 'expense';
    }
  }

  // Default to expense if amount detected but type unclear
  return 'expense';
}

/**
 * Detect category from SMS text
 */
function detectCategory(text: string, type: 'income' | 'expense' | 'transfer'): string {
  // Check for salary first for income
  if (type === 'income' && /salary|wages|payroll/i.test(text)) {
    return 'salary';
  }

  // Check each category pattern
  for (const { pattern, category } of CATEGORY_PATTERNS) {
    if (pattern.test(text)) {
      return category;
    }
  }

  // Default categories based on type
  if (type === 'income') return 'other_income';
  if (type === 'expense') return 'other_expense';
  return 'transfer';
}

/**
 * Check if SMS is a transaction message
 */
export function isTransactionSMS(sms: SMSMessage): boolean {
  const sender = sms.address.toUpperCase();
  const body = sms.body.toLowerCase();

  // Check if sender is a known bank/financial service
  const isFinancialSender = BANK_SENDERS.some(bank => 
    sender.includes(bank) || sender.includes(bank.toLowerCase())
  );

  // Check for transaction keywords
  const hasTransactionKeywords = 
    [...DEBIT_KEYWORDS, ...CREDIT_KEYWORDS].some(kw => body.includes(kw)) ||
    UPI_KEYWORDS.some(kw => body.toLowerCase().includes(kw.toLowerCase()));

  // Check for amount pattern
  const hasAmount = /(?:rs\.?|inr|₹)\s*[\d,]+/i.test(body);

  return (isFinancialSender || hasTransactionKeywords) && hasAmount;
}

/**
 * Parse a single SMS message into transaction data
 */
export function parseSMS(sms: SMSMessage): ParsedTransaction | null {
  if (!isTransactionSMS(sms)) {
    return null;
  }

  const body = sms.body;
  const amount = parseAmount(body);

  if (!amount) {
    return null;
  }

  const type = detectTransactionType(body);
  const category = detectCategory(body, type);
  const merchant = parseMerchant(body);
  const balance = parseBalance(body);
  const accountNumber = parseAccountNumber(body);
  const referenceId = parseReferenceId(body);

  // Create description from SMS content
  let description = merchant || category;
  if (referenceId) {
    description += ` (Ref: ${referenceId})`;
  }

  return {
    type,
    amount,
    category,
    description,
    date: new Date(sms.date),
    recipient: type === 'expense' ? merchant || undefined : undefined,
    merchantName: merchant || undefined,
    accountNumber: accountNumber || undefined,
    balance: balance || undefined,
    referenceId: referenceId || undefined,
  };
}

/**
 * Parse multiple SMS messages and return valid transactions
 */
export function parseMultipleSMS(messages: SMSMessage[]): ParsedTransaction[] {
  const transactions: ParsedTransaction[] = [];

  for (const sms of messages) {
    const parsed = parseSMS(sms);
    if (parsed) {
      transactions.push(parsed);
    }
  }

  // Sort by date, newest first
  transactions.sort((a, b) => b.date.getTime() - a.date.getTime());

  // Remove duplicates based on amount, date, and type
  const unique = transactions.filter((txn, index, self) => {
    return index === self.findIndex(t => 
      t.amount === txn.amount &&
      t.type === txn.type &&
      Math.abs(t.date.getTime() - txn.date.getTime()) < 60000 // Within 1 minute
    );
  });

  return unique;
}

/**
 * Get category icon for a transaction category
 */
export function getCategoryIcon(category: string): string {
  const icons: Record<string, string> = {
    food: '🍔',
    shopping: '🛒',
    transport: '🚗',
    utilities: '💡',
    entertainment: '🎬',
    healthcare: '🏥',
    education: '📚',
    groceries: '🥬',
    insurance: '🛡️',
    investment: '📈',
    rent: '🏠',
    cash: '💵',
    transfer: '↔️',
    salary: '💰',
    other_income: '💵',
    other_expense: '💸',
  };

  return icons[category] || '💳';
}
