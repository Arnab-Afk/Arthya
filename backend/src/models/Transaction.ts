import { Table, Column, Model, DataType, ForeignKey, BelongsTo } from 'sequelize-typescript';
import User from './User';

@Table({
  tableName: 'transactions',
  timestamps: true,
  indexes: [
    { fields: ['userId', 'date'] },
    { fields: ['userId', 'type'] },
    { fields: ['userId', 'category'] },
  ],
})
export default class Transaction extends Model {
  @ForeignKey(() => User)
  @Column({
    type: DataType.INTEGER,
    allowNull: false,
  })
  userId!: number;

  @BelongsTo(() => User)
  user!: User;

  @Column({
    type: DataType.ENUM('income', 'expense', 'transfer'),
    allowNull: false,
  })
  type!: 'income' | 'expense' | 'transfer';

  @Column({
    type: DataType.STRING,
    allowNull: false,
  })
  category!: string;

  @Column({
    type: DataType.DECIMAL(10, 2),
    allowNull: false,
    validate: {
      min: 0,
    },
  })
  amount!: number;

  @Column({
    type: DataType.TEXT,
    allowNull: true,
  })
  description?: string;

  @Column({
    type: DataType.DATE,
    defaultValue: DataType.NOW,
  })
  date!: Date;

  @Column({
    type: DataType.STRING,
    allowNull: true,
  })
  recipient?: string;

  @Column({
    type: DataType.ENUM('completed', 'pending', 'failed'),
    defaultValue: 'completed',
  })
  status!: 'completed' | 'pending' | 'failed';

  @Column({
    type: DataType.STRING,
    allowNull: true,
    defaultValue: 'manual',
  })
  source?: 'manual' | 'sms_import' | 'bank_sync' | 'upi';

  @Column({
    type: DataType.TEXT,
    allowNull: true,
  })
  notes?: string;

  @Column({
    type: DataType.JSONB,
    allowNull: true,
  })
  metadata?: {
    location?: string;
    paymentMethod?: string;
    merchantName?: string;
    bankName?: string;
    smsBody?: string;
  };
}
