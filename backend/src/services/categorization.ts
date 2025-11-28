/**
 * Smart Expense Categorization Service
 * Automatically categorizes transactions based on description and patterns
 */

interface CategoryRule {
  pattern: RegExp;
  category: string;
  subCategory?: string;
}

interface CategorizationResult {
  category: string;
  subCategory?: string;
  confidence: number;
  suggestedCategories?: string[];
}

// Category definitions with icons
export const EXPENSE_CATEGORIES = {
  food: {
    name: 'Food & Dining',
    icon: '🍔',
    subCategories: ['restaurants', 'groceries', 'food_delivery', 'coffee', 'fast_food'],
  },
  transport: {
    name: 'Transportation',
    icon: '🚗',
    subCategories: ['fuel', 'public_transport', 'ride_share', 'parking', 'vehicle_maintenance', 'tolls'],
  },
  shopping: {
    name: 'Shopping',
    icon: '🛒',
    subCategories: ['clothing', 'electronics', 'home_goods', 'personal_care', 'gifts'],
  },
  utilities: {
    name: 'Utilities & Bills',
    icon: '💡',
    subCategories: ['electricity', 'water', 'gas', 'internet', 'mobile', 'dth'],
  },
  entertainment: {
    name: 'Entertainment',
    icon: '🎬',
    subCategories: ['movies', 'streaming', 'gaming', 'sports', 'events', 'hobbies'],
  },
  healthcare: {
    name: 'Healthcare',
    icon: '🏥',
    subCategories: ['doctor', 'pharmacy', 'hospital', 'insurance_health', 'fitness'],
  },
  education: {
    name: 'Education',
    icon: '📚',
    subCategories: ['tuition', 'books', 'courses', 'school_fees', 'coaching'],
  },
  rent: {
    name: 'Rent & Housing',
    icon: '🏠',
    subCategories: ['rent', 'maintenance', 'repairs', 'home_improvement'],
  },
  insurance: {
    name: 'Insurance',
    icon: '🛡️',
    subCategories: ['life_insurance', 'health_insurance', 'vehicle_insurance', 'home_insurance'],
  },
  investment: {
    name: 'Investments',
    icon: '📈',
    subCategories: ['mutual_funds', 'stocks', 'sip', 'fd', 'gold', 'crypto'],
  },
  personal: {
    name: 'Personal Care',
    icon: '💇',
    subCategories: ['salon', 'spa', 'grooming', 'cosmetics'],
  },
  travel: {
    name: 'Travel',
    icon: '✈️',
    subCategories: ['flights', 'hotels', 'trains', 'vacation'],
  },
  subscriptions: {
    name: 'Subscriptions',
    icon: '📱',
    subCategories: ['streaming', 'software', 'magazines', 'memberships'],
  },
  cash: {
    name: 'Cash & ATM',
    icon: '💵',
    subCategories: ['atm_withdrawal', 'cash_payment'],
  },
  transfer: {
    name: 'Transfers',
    icon: '↔️',
    subCategories: ['upi', 'neft', 'imps', 'personal_transfer'],
  },
  other: {
    name: 'Other',
    icon: '📦',
    subCategories: ['miscellaneous', 'uncategorized'],
  },
};

export const INCOME_CATEGORIES = {
  salary: {
    name: 'Salary',
    icon: '💰',
    subCategories: ['monthly_salary', 'bonus', 'arrears'],
  },
  freelance: {
    name: 'Freelance',
    icon: '💻',
    subCategories: ['project_payment', 'consulting', 'gig_work'],
  },
  business: {
    name: 'Business Income',
    icon: '🏪',
    subCategories: ['sales', 'services', 'commission'],
  },
  investment_income: {
    name: 'Investment Returns',
    icon: '📊',
    subCategories: ['dividends', 'interest', 'capital_gains', 'rental_income'],
  },
  refund: {
    name: 'Refunds',
    icon: '↩️',
    subCategories: ['shopping_refund', 'tax_refund', 'insurance_claim'],
  },
  cashback: {
    name: 'Cashback & Rewards',
    icon: '🎁',
    subCategories: ['cashback', 'rewards', 'points_redemption'],
  },
  other_income: {
    name: 'Other Income',
    icon: '💵',
    subCategories: ['gifts', 'miscellaneous'],
  },
};

// Categorization rules based on keywords
const EXPENSE_RULES: CategoryRule[] = [
  // Food & Dining
  { pattern: /swiggy|zomato|food|restaurant|cafe|hotel|dining|dominos|pizza|mcdonald|kfc|burger|subway|starbucks|dunkin|biryani|meals|lunch|dinner|breakfast/i, category: 'food', subCategory: 'restaurants' },
  { pattern: /bigbasket|grofers|blinkit|zepto|instamart|dmart|reliance fresh|more supermarket|grocery|vegetables|fruits|supermarket/i, category: 'food', subCategory: 'groceries' },
  { pattern: /uber eats|doordash|deliveroo/i, category: 'food', subCategory: 'food_delivery' },
  
  // Transport
  { pattern: /uber|ola|rapido|ride|cab|taxi/i, category: 'transport', subCategory: 'ride_share' },
  { pattern: /petrol|diesel|fuel|gas station|hp|indian oil|bharat petroleum|shell/i, category: 'transport', subCategory: 'fuel' },
  { pattern: /metro|bus|train ticket|railway|irctc|mmts|local train/i, category: 'transport', subCategory: 'public_transport' },
  { pattern: /parking|fastag|toll/i, category: 'transport', subCategory: 'parking' },
  { pattern: /service center|car wash|vehicle|repair|mechanic/i, category: 'transport', subCategory: 'vehicle_maintenance' },
  
  // Shopping
  { pattern: /amazon|flipkart|myntra|ajio|meesho|shopping|mall|retail|store|mart|bazaar|market|shoppers stop|lifestyle/i, category: 'shopping' },
  { pattern: /clothing|apparel|fashion|shirt|jeans|dress|wear/i, category: 'shopping', subCategory: 'clothing' },
  { pattern: /electronics|mobile|laptop|phone|tablet|computer|gadget|croma|reliance digital/i, category: 'shopping', subCategory: 'electronics' },
  
  // Utilities
  { pattern: /electricity|power|bescom|tata power|adani|electric bill/i, category: 'utilities', subCategory: 'electricity' },
  { pattern: /water bill|water supply|bwssb/i, category: 'utilities', subCategory: 'water' },
  { pattern: /piped gas|hp gas|indane|bharat gas|lpg|cooking gas/i, category: 'utilities', subCategory: 'gas' },
  { pattern: /internet|broadband|wifi|jio fiber|airtel xstream|act fibernet/i, category: 'utilities', subCategory: 'internet' },
  { pattern: /mobile recharge|prepaid|postpaid|airtel|jio|vi|vodafone|idea|bsnl/i, category: 'utilities', subCategory: 'mobile' },
  { pattern: /dth|tata sky|dish tv|sun direct|d2h|airtel digital/i, category: 'utilities', subCategory: 'dth' },
  
  // Entertainment
  { pattern: /netflix|prime video|hotstar|disney|zee5|sonyliv|streaming/i, category: 'entertainment', subCategory: 'streaming' },
  { pattern: /movie|cinema|pvr|inox|bookmyshow|film/i, category: 'entertainment', subCategory: 'movies' },
  { pattern: /spotify|gaana|wynk|apple music|youtube music/i, category: 'entertainment', subCategory: 'streaming' },
  { pattern: /gaming|playstation|xbox|steam|epic games|pubg|freefire/i, category: 'entertainment', subCategory: 'gaming' },
  
  // Healthcare
  { pattern: /hospital|clinic|doctor|consultation|medical|health|apollo|fortis|manipal|max hospital/i, category: 'healthcare', subCategory: 'doctor' },
  { pattern: /pharmacy|medicine|drug|medplus|apollo pharmacy|netmeds|1mg|pharmeasy/i, category: 'healthcare', subCategory: 'pharmacy' },
  { pattern: /gym|fitness|yoga|cult|cult.fit|healthify|workout/i, category: 'healthcare', subCategory: 'fitness' },
  { pattern: /diagnostic|lab|test|thyrocare|lal path|metropolis/i, category: 'healthcare', subCategory: 'hospital' },
  
  // Education
  { pattern: /school|college|university|tuition|coaching|institute|academy/i, category: 'education', subCategory: 'tuition' },
  { pattern: /course|udemy|coursera|edx|skillshare|masterclass|upgrad|byju|unacademy/i, category: 'education', subCategory: 'courses' },
  { pattern: /book|stationary|notebook|amazon kindle|audible/i, category: 'education', subCategory: 'books' },
  
  // Rent & Housing
  { pattern: /rent|landlord|tenant|housing|apartment|flat|pg|hostel/i, category: 'rent', subCategory: 'rent' },
  { pattern: /society|maintenance|association/i, category: 'rent', subCategory: 'maintenance' },
  { pattern: /plumber|electrician|carpenter|home repair|renovation/i, category: 'rent', subCategory: 'repairs' },
  
  // Insurance
  { pattern: /lic|life insurance|term insurance|hdfc life|icici pru|sbi life|max life/i, category: 'insurance', subCategory: 'life_insurance' },
  { pattern: /health insurance|mediclaim|star health|care health|niva bupa/i, category: 'insurance', subCategory: 'health_insurance' },
  { pattern: /car insurance|vehicle insurance|motor insurance|bike insurance/i, category: 'insurance', subCategory: 'vehicle_insurance' },
  
  // Investment
  { pattern: /mutual fund|mf|sip|zerodha|groww|upstox|paytm money|kuvera|et money/i, category: 'investment', subCategory: 'mutual_funds' },
  { pattern: /stock|share|equity|nse|bse|demat/i, category: 'investment', subCategory: 'stocks' },
  { pattern: /fd|fixed deposit|recurring deposit|rd/i, category: 'investment', subCategory: 'fd' },
  
  // Personal Care
  { pattern: /salon|spa|parlour|beauty|grooming|haircut|facial/i, category: 'personal', subCategory: 'salon' },
  { pattern: /cosmetic|makeup|skincare|nykaa|purplle/i, category: 'personal', subCategory: 'cosmetics' },
  
  // Travel
  { pattern: /flight|airline|indigo|spicejet|air india|vistara|goair|makemytrip|cleartrip|yatra/i, category: 'travel', subCategory: 'flights' },
  { pattern: /hotel|oyo|goibibo|booking\.com|airbnb|treebo|fabhotel/i, category: 'travel', subCategory: 'hotels' },
  
  // Subscriptions
  { pattern: /subscription|membership|premium|annual|monthly plan/i, category: 'subscriptions' },
  
  // Cash & ATM
  { pattern: /atm|cash withdrawal|atm withdrawal|cash deposit/i, category: 'cash', subCategory: 'atm_withdrawal' },
  
  // Transfer
  { pattern: /transfer|upi|neft|imps|rtgs|sent to|paid to/i, category: 'transfer', subCategory: 'upi' },
];

const INCOME_RULES: CategoryRule[] = [
  { pattern: /salary|wages|payroll|monthly pay/i, category: 'salary', subCategory: 'monthly_salary' },
  { pattern: /bonus|incentive|variable pay/i, category: 'salary', subCategory: 'bonus' },
  { pattern: /freelance|project|consulting|client payment|gig|contract/i, category: 'freelance', subCategory: 'project_payment' },
  { pattern: /business|sales|revenue|commission|earnings/i, category: 'business', subCategory: 'sales' },
  { pattern: /dividend|interest|fd interest|savings interest/i, category: 'investment_income', subCategory: 'interest' },
  { pattern: /rent received|rental income|tenant/i, category: 'investment_income', subCategory: 'rental_income' },
  { pattern: /refund|reimbursement|returned/i, category: 'refund', subCategory: 'shopping_refund' },
  { pattern: /cashback|reward|points/i, category: 'cashback', subCategory: 'cashback' },
];

/**
 * Categorize a transaction based on description
 */
export function categorizeTransaction(
  description: string,
  merchantName?: string,
  type: 'income' | 'expense' | 'transfer' = 'expense'
): CategorizationResult {
  const searchText = `${description} ${merchantName || ''}`.toLowerCase();
  
  const rules = type === 'income' ? INCOME_RULES : EXPENSE_RULES;
  
  // Try to match against rules
  for (const rule of rules) {
    if (rule.pattern.test(searchText)) {
      return {
        category: rule.category,
        subCategory: rule.subCategory,
        confidence: 85,
      };
    }
  }

  // If no match found, return other category with low confidence
  const defaultCategory = type === 'income' ? 'other_income' : 'other';
  return {
    category: defaultCategory,
    confidence: 30,
    suggestedCategories: type === 'income' 
      ? ['salary', 'freelance', 'business', 'refund']
      : ['food', 'shopping', 'transport', 'utilities'],
  };
}

/**
 * Batch categorize multiple transactions
 */
export function batchCategorize(
  transactions: Array<{
    id: string;
    description: string;
    merchantName?: string;
    type: 'income' | 'expense' | 'transfer';
  }>
): Array<{ id: string } & CategorizationResult> {
  return transactions.map(txn => ({
    id: txn.id,
    ...categorizeTransaction(txn.description, txn.merchantName, txn.type),
  }));
}

/**
 * Get category details
 */
export function getCategoryDetails(category: string, type: 'income' | 'expense' = 'expense'): {
  name: string;
  icon: string;
  subCategories: string[];
} | null {
  const categories = type === 'income' ? INCOME_CATEGORIES : EXPENSE_CATEGORIES;
  const categoryData = (categories as any)[category];
  
  if (!categoryData) return null;
  
  return {
    name: categoryData.name,
    icon: categoryData.icon,
    subCategories: categoryData.subCategories,
  };
}

/**
 * Get all categories
 */
export function getAllCategories(type: 'income' | 'expense' = 'expense'): Array<{
  key: string;
  name: string;
  icon: string;
}> {
  const categories = type === 'income' ? INCOME_CATEGORIES : EXPENSE_CATEGORIES;
  
  return Object.entries(categories).map(([key, value]) => ({
    key,
    name: value.name,
    icon: value.icon,
  }));
}

/**
 * Detect recurring transactions (subscriptions)
 */
export function detectRecurringPattern(
  transactions: Array<{
    amount: number;
    description: string;
    merchantName?: string;
    date: Date;
  }>
): Array<{
  pattern: string;
  amount: number;
  frequency: 'daily' | 'weekly' | 'monthly' | 'yearly';
  nextExpectedDate: Date;
  confidence: number;
}> {
  const recurring: Array<{
    pattern: string;
    amount: number;
    frequency: 'daily' | 'weekly' | 'monthly' | 'yearly';
    nextExpectedDate: Date;
    confidence: number;
  }> = [];

  // Group by similar amount and description
  const grouped: { [key: string]: typeof transactions } = {};
  
  transactions.forEach(txn => {
    const key = `${txn.description.toLowerCase().substring(0, 20)}_${Math.round(txn.amount)}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(txn);
  });

  // Analyze each group for recurring patterns
  for (const [_key, group] of Object.entries(grouped)) {
    if (group.length < 2) continue;

    const sorted = group.sort((a, b) => a.date.getTime() - b.date.getTime());
    const intervals: number[] = [];

    for (let i = 1; i < sorted.length; i++) {
      const daysDiff = Math.round(
        (sorted[i].date.getTime() - sorted[i-1].date.getTime()) / (1000 * 60 * 60 * 24)
      );
      intervals.push(daysDiff);
    }

    if (intervals.length === 0) continue;

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((sum, i) => sum + Math.pow(i - avgInterval, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);

    // If consistent interval (low variance), it's likely recurring
    if (stdDev < avgInterval * 0.3) {
      let frequency: 'daily' | 'weekly' | 'monthly' | 'yearly';
      
      if (avgInterval <= 2) frequency = 'daily';
      else if (avgInterval <= 10) frequency = 'weekly';
      else if (avgInterval <= 35) frequency = 'monthly';
      else frequency = 'yearly';

      const lastDate = sorted[sorted.length - 1].date;
      const nextExpectedDate = new Date(lastDate);
      nextExpectedDate.setDate(nextExpectedDate.getDate() + Math.round(avgInterval));

      recurring.push({
        pattern: group[0].description,
        amount: group[0].amount,
        frequency,
        nextExpectedDate,
        confidence: Math.max(50, 100 - stdDev * 5),
      });
    }
  }

  return recurring;
}
