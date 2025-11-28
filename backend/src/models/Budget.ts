import { Table, Column, Model, DataType, ForeignKey, BelongsTo } from 'sequelize-typescript';
import User from './User';

@Table({
  tableName: 'budgets',
  timestamps: true,
})
export default class Budget extends Model {
  @ForeignKey(() => User)
  @Column({
    type: DataType.INTEGER,
    allowNull: false,
  })
  userId!: number;

  @BelongsTo(() => User)
  user!: User;

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
    type: DataType.DECIMAL(10, 2),
    defaultValue: 0,
  })
  spent!: number;

  @Column({
    type: DataType.ENUM('weekly', 'monthly', 'yearly'),
    defaultValue: 'monthly',
  })
  period!: 'weekly' | 'monthly' | 'yearly';

  @Column({
    type: DataType.DATE,
    allowNull: false,
  })
  startDate!: Date;

  @Column({
    type: DataType.DATE,
    allowNull: true,
  })
  endDate?: Date;

  @Column({
    type: DataType.BOOLEAN,
    defaultValue: true,
  })
  isActive!: boolean;

  @Column({
    type: DataType.BOOLEAN,
    defaultValue: true,
  })
  alertsEnabled!: boolean;

  @Column({
    type: DataType.INTEGER,
    defaultValue: 80,
    comment: 'Alert when spending reaches this percentage',
  })
  alertThreshold!: number;

  @Column({
    type: DataType.STRING,
    allowNull: true,
  })
  notes?: string;
}
